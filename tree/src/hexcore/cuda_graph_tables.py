"""Retained device descriptor tables prepared before CUDA graph capture."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable


class GraphTables:
    """Keep descriptor uploads out of capture and its scratch allocator.

    Each owner keeps its normal bounded table cache. This controller retains
    every table used by one graph, including tables that cache later evicts.
    Its instance hooks exist only while preparing or recording that graph.
    """

    def __init__(self, cp: Any, objects: Iterable[Any], allocator: Any) -> None:
        self.cp = cp
        self.allocator = allocator
        self.objects = tuple({id(obj): obj for obj in objects if obj is not None}.values())
        self.capturing = False
        self._tables: dict[tuple[int, tuple[tuple[int, ...], ...]], Any] = {}
        self._uploads = 0
        self._uploads_measured = bool(self.objects) and all(
            hasattr(obj, "table_uploads") for obj in self.objects
        )
        self._depth = 0

    def _table(self, obj: Any, original: Any, rows: Any) -> Any:
        rows_key = tuple(tuple(int(word) for word in row) for row in rows)
        key = (id(obj), rows_key)
        found = self._tables.get(key)
        if found is not None:
            return found
        if self.capturing:
            raise RuntimeError(
                "CUDA graph descriptor table was not prepared; uploading it "
                "during capture would perform a host-to-device transfer"
            )
        before = getattr(obj, "table_uploads", None)
        # Descriptor allocations occur only in preparation. Recording them as
        # scratch would make capture's allocation sequence differ on a hit.
        with self.cp.cuda.using_allocator(self.allocator):
            found = original(rows_key)
        after = getattr(obj, "table_uploads", None)
        if before is not None and after is not None:
            self._uploads += int(after) - int(before)
        self._tables[key] = found
        return found

    @contextmanager
    def scope(self):
        """Hook only the supplied instances and restore them on every exit."""
        if self._depth:
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
            return
        restored: list[tuple[Any, bool, Any]] = []
        self._depth = 1
        try:
            for obj in self.objects:
                namespace = vars(obj)
                owned = "table" in namespace
                previous = namespace.get("table")
                original = obj.table

                def table(rows, owner=obj, method=original):
                    return self._table(owner, method, rows)

                obj.table = table
                restored.append((obj, owned, previous))
            yield self
        finally:
            for obj, owned, previous in reversed(restored):
                if owned:
                    obj.table = previous
                else:
                    del obj.table
            self._depth = 0

    def receipt(self) -> dict[str, Any]:
        tables = {id(table): table for table in self._tables.values()}
        return {
            "tables": len(self._tables),
            "uploads_during_prepare": self._uploads if self._uploads_measured else None,
            "retained_table_bytes": sum(int(table.nbytes) for table in tables.values()),
        }


__all__ = ["GraphTables"]
