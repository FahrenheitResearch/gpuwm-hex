"""The MPAS render door: history files in, product PNGs out, Rust path only.

``mpas-port render`` is the user-reachable command over two Rust binaries:

* ``rw_mpas_convert`` (gpuwm ``tools/rustwx/crates/rw-mpas``) resamples one or
  more MPAS history frames onto a named render window and writes
  wrfout-shaped netCDF frames, with a JSON receipt carrying the mesh,
  weights and per-frame output digests.
* ``rw_wrfbatch`` (gpuwm ``tools/rustwx/crates/rw-wrfbatch``) imports each
  frame and draws the product PNGs.  Product availability is proven by the
  renderer's own ``--list-products`` catalog against the stored fields,
  never guessed from filenames.

This module is deliberately subprocess-and-stdlib only: every byte of field
data is decoded, resampled and drawn in Rust.  Python here is orchestration --
resolving executables, sequencing the two engines, filing the finished PNGs
into the ruled ``<domain>/<product>/<valid-day>/`` layout and writing the
manifest.  There is no fallback plotter behind it.

Scratch discipline: conversion frames, per-frame render stores and the raw
engine output live in a scratch directory that is a SIBLING of ``--out``,
never inside it.  A ``.rwstore-*`` scratch tree inside a delivered png tree
is a shipped defect this door refuses to repeat; an explicit ``--scratch``
inside ``--out`` is refused by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from . import engines
from .errors import MpasPortError


class RenderDoorError(MpasPortError):
    """A named render-door refusal: what breaks, and the remedy."""


#: Environment variables naming the two engine binaries.  The renderer also
#: honors gpuwm's own spelling so a box with a gpuwm install needs no second
#: variable pointed at the same file.
# The preferred spellings carry the distribution's own name.  The
# MPAS_PORT_* pair came from the import package's ORIGINAL name -- the
# package was `mpas_port` through 0.1.1 and is `hexcore` from 0.2.0 -- and it
# predates the decision that the brand and the spellings a user types never
# carry the MPAS token.  They are kept AHEAD of nothing and BEHIND the new
# names, never removed: an install line that already works must not be
# invalidated by a rename, and a variable that silently stops being read is
# the worst possible way to learn about one.  That rule is why the 0.2.0
# package rename did not touch this pair either: these two strings are the
# one place the retired name is still load-bearing for a user.
#
# The spellings are stated here and CONSUMED by :mod:`hexcore.engines`,
# which owns the one resolution ladder every door in this distribution
# uses.  Both halves matter: the names live beside the door that
# documents them, and the search itself is not reimplemented per door --
# which is how this door came to read three variables and PATH while the
# init door read one variable and PATH, and neither read the directory
# `gpuwm fetch-bridges` actually stages into.
CONVERT_ENV = "GPUWM_HEX_RW_MPAS_CONVERT"
CONVERT_ENV_LEGACY = "MPAS_PORT_RW_MPAS_CONVERT"
RENDERER_ENV = "GPUWM_HEX_RW_WRFBATCH"
RENDERER_ENV_LEGACY = "MPAS_PORT_RW_WRFBATCH"
RENDERER_ENV_GPUWM = "GPUWM_RW_WRFBATCH"

# The ladder reads them in this order, preferred first, legacy behind,
# gpuwm's own spelling last of the three.  A test binds these tuples to
# the constants above so a rename cannot drop a variable from the search.
assert engines.CONVERT.env_names == (CONVERT_ENV, CONVERT_ENV_LEGACY)
assert engines.RENDERER.env_names == (RENDERER_ENV, RENDERER_ENV_LEGACY)
assert engines.RENDERER.gpuwm_env == RENDERER_ENV_GPUWM

#: The manifest schema written beside the delivered tree.
MANIFEST_SCHEMA = "mpas-port.render-door-manifest/v1"

#: Layout tokens for a frame whose facts could not be read (same spellings
#: as gpuwm.render_layout, so a reader of either tree learns one vocabulary).
NATIVE_GRID = "native_grid"
UNDATED = "undated"
UNCLASSIFIED = "unclassified"

#: ``--products`` tokens the renderer expands itself; they name groups, not
#: single catalog rows, so per-slug coverage refusal does not apply to them.
_GROUP_KEYWORDS = frozenset({"all", "direct", "derived", "heavy", "windowed"})

#: The renderer's output filename, as rustwx-products formats it::
#:
#:     rustwx_<model>_<YYYYMMDD>_<H>z_f<NNN>_<domain-slug>_<product-slug>.png
_LEAD_SEGMENT = re.compile(r"^f\d{3}$")

#: ``YYYY-MM-DD`` at the head of an ISO instant or WRF Times stamp.
_DAY = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T_ ]|$)")


@dataclass
class FrameResult:
    """What one history frame became."""

    history: str
    wrfout: str
    valid_time: str
    valid_day: str
    output_sha256: str
    absent_wrf_fields: list[str]
    rendered: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    placed: list[str] = field(default_factory=list)
    # Composed frames only: how far the runs disagree where the composite
    # changes source, per surface field.  Empty for a single-run render,
    # which has no seam to measure.
    seam: list[dict] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(spec: engines.EngineSpec, explicit: str | None) -> Path:
    """Drive the shared ladder, re-raising as this door's own refusal type.

    The refusal TEXT is the ladder's, unedited: it already names every
    rung it searched and both commands that supply the binary, and a
    second paraphrase here would be a second thing to keep true.  Only
    the exception CLASS changes, so a caller catching
    :class:`RenderDoorError` still catches it.
    """

    try:
        return engines.resolve(spec, explicit)
    except engines.EngineRefusal as error:
        raise RenderDoorError(str(error)) from None


def resolve_convert_exe(explicit: str | None = None) -> Path:
    """The ``rw_mpas_convert`` binary, or a named refusal.

    Reads ``--convert-exe``, ``$CONVERT_ENV``, ``$CONVERT_ENV_LEGACY``,
    gpuwm's ``$GPUWM_RW_MPAS_CONVERT`` and its bridge directories, then
    ``PATH``.
    """

    return _resolve(engines.CONVERT, explicit)


def resolve_renderer_exe(explicit: str | None = None) -> Path:
    """The ``rw_wrfbatch`` binary, or a named refusal.

    Reads ``--renderer-exe``, ``$RENDERER_ENV``, ``$RENDERER_ENV_LEGACY``,
    ``$RENDERER_ENV_GPUWM`` and gpuwm's bridge directories, then ``PATH``.
    """

    return _resolve(engines.RENDERER, explicit)


def valid_day(stamp: str | None) -> str:
    """``YYYY-MM-DD`` from an ISO instant, or :data:`UNDATED` -- a wrong
    date is worse than an honest ``undated`` because a reader believes it."""

    if not stamp:
        return UNDATED
    match = _DAY.match(str(stamp).strip())
    if match is None:
        return UNDATED
    try:
        _dt.date.fromisoformat(match.group(1))
    except ValueError:
        return UNDATED
    return match.group(1)


def split_engine_name(filename: str, slug: str) -> tuple[str, str]:
    """``(domain_token, product_dir)`` for one engine output filename.

    The RENDERED event names the product ``slug``, but the FILENAME spells
    it the engine's filesystem-safe way: a generic ``var:wrf_psfc`` event
    becomes ``..._var_wrf_psfc_<hash>.png`` on disk (colon to underscore,
    plus a content-hash suffix).  The product directory uses that safe
    spelling too -- a ``var:`` directory is unrepresentable on Windows and
    would strand the delivered tree on the render node.  The domain token
    is what sits between the ``fNNN`` lead segment and the slug's first
    appearance.  A name that does not parse files under
    (:data:`NATIVE_GRID`, safe slug) -- somewhere nameable, never dropped.
    """

    safe_slug = slug.replace(":", "_")
    stem = Path(filename).stem
    parts = stem.split("_")
    for index, part in enumerate(parts):
        if _LEAD_SEGMENT.match(part):
            tail = "_".join(parts[index + 1 :])
            marker = tail.find(f"_{safe_slug}")
            if marker > 0:
                return tail[:marker], safe_slug
            return NATIVE_GRID, safe_slug
    return NATIVE_GRID, safe_slug


def place_path(out_root: Path, domain: str, product: str, day: str,
               filename: str) -> Path:
    """``<out>/<domain>/<product>/<valid-day>/<filename>`` -- the ruled
    layout, and the only one this door writes.  Nothing may invert it back
    to flat."""

    return (Path(out_root) / (domain or NATIVE_GRID)
            / (product or UNCLASSIFIED) / (day or UNDATED) / filename)


def default_scratch(out_root: Path) -> Path:
    """A scratch sibling of ``--out``: same parent, never inside the tree."""

    out_root = Path(out_root)
    return out_root.parent / (out_root.name + ".render-scratch")


def refuse_scratch_inside_out(scratch: Path, out_root: Path) -> None:
    scratch_abs = Path(os.path.abspath(scratch))
    out_abs = Path(os.path.abspath(out_root))
    if scratch_abs == out_abs or out_abs in scratch_abs.parents:
        raise RenderDoorError(
            f"--scratch {scratch} is inside the delivered png tree "
            f"{out_root}. Renderer scratch inside a delivered tree is a "
            f"shipped defect (readers publish half-finished temporaries as "
            f"products); give --scratch a directory outside --out, or omit "
            f"it for the sibling default."
        )


def _require_file(path: Path, flag: str, remedy: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise RenderDoorError(
            f"{flag} names a missing file: {path}. {remedy}"
        )
    return path


def _run(command: list[str], *, log_prefix: str) -> subprocess.CompletedProcess:
    printable = " ".join(str(part) for part in command)
    print(f"{log_prefix} $ {printable}", flush=True)
    return subprocess.run(
        [str(part) for part in command],
        capture_output=True, text=True, errors="replace",
    )


def convert_frames(convert_exe: Path, *, histories: list[Path] | None,
                   mesh: Path | None, scratch: Path, window: str,
                   field_set: str, simulation_start: str | None,
                   nc_format: str, compose: Path | None = None,
                   compose_base_only: bool = False) -> tuple[dict, list[str]]:
    """Run ``rw_mpas_convert`` once over the whole series.

    Returns ``(report, invocation)`` where ``report`` is the converter's own
    JSON receipt (mesh/weights digests, per-frame output digests and absent
    WRF fields).  A converter refusal is relayed verbatim -- it already
    names the file and the field.

    With ``compose``, the source list in that file owns the meshes and the
    histories and ``histories``/``mesh`` are not passed: two source lists
    that can disagree about which run is the base is a worse failure than
    one flag being unavailable, so the converter refuses the pair and this
    door does not construct it."""

    frames_dir = Path(scratch) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(scratch) / "convert-report.json"
    command: list[str] = [str(convert_exe)]
    if compose is not None:
        command.extend(("--compose", str(compose)))
        if compose_base_only:
            command.append("--compose-base-only")
    else:
        command.extend(("--history", *[str(item) for item in histories or []]))
        command.extend(("--mesh", str(mesh)))
    command.extend((
        "--out-dir", str(frames_dir),
        "--window", window,
        "--field-set", field_set,
        "--format", nc_format,
        "--clobber",
        "--json", str(report_path),
    ))
    if simulation_start:
        command.extend(("--simulation-start", simulation_start))
    result = _run(command, log_prefix="CONVERT")
    for line in (result.stdout or "").splitlines():
        if line.startswith(("WINDOW\t", "WEIGHTS\t", "CONVERTED\t",
                            "COMPOSITE\t", "SEAM\t", "FINISHED\t")):
            print(f"CONVERT {line}", flush=True)
    # A composite drops any hour a source does not carry, and says so on
    # stderr.  Relayed rather than swallowed: a frame silently missing from
    # a delivered series reads as a gap in the weather.
    for line in (result.stderr or "").splitlines():
        if "dropping" in line:
            print(f"CONVERT {line.strip()}", flush=True)
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        raise RenderDoorError(
            f"rw_mpas_convert exited {result.returncode}: "
            f"{tail[-1] if tail else 'no stderr'}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RenderDoorError(
            f"rw_mpas_convert exited 0 but its JSON receipt at "
            f"{report_path} is unreadable ({error}); the receipt is the "
            f"digest evidence, so the render does not proceed without it."
        ) from error
    return report, [str(part) for part in command]


def list_catalog(renderer_exe: Path, wrfout: Path, *, scratch: Path,
                 heavy: bool) -> tuple[dict[str, tuple[str, str, str]], str,
                                       list[str]]:
    """The renderer's own product catalog for one converted frame.

    Returns ``(rows, summary, invocation)``; ``rows`` maps slug ->
    (kind, status, detail).  Statuses are proven against the real stored
    import, never guessed."""

    store = Path(scratch) / "catalog-store"
    store.mkdir(parents=True, exist_ok=True)
    command = [
        str(renderer_exe),
        "--store-root", str(store),
        "--out-dir", str(store),
        "--list-products",
    ]
    if heavy:
        command.append("--heavy")
    command.append(str(wrfout))
    result = _run(command, log_prefix="CATALOG")
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        raise RenderDoorError(
            f"rw_wrfbatch --list-products exited {result.returncode} on "
            f"{wrfout}: {tail[-1] if tail else 'no stderr'}"
        )
    rows: dict[str, tuple[str, str, str]] = {}
    summary = ""
    for line in (result.stdout or "").splitlines():
        if line.startswith("PRODUCT\t"):
            parts = line.split("\t")
            if len(parts) == 5:
                rows[parts[1]] = (parts[2], parts[3], parts[4])
        elif line.startswith("CATALOG "):
            summary = line[len("CATALOG "):]
    if not rows:
        raise RenderDoorError(
            f"rw_wrfbatch produced no catalog rows for {wrfout}; the import "
            f"failed silently, so product availability cannot be proven and "
            f"the render does not proceed."
        )
    return rows, summary, [str(part) for part in command]


def refuse_uncovered_products(products: str, rows: dict, history: Path,
                              absent: list[str]) -> None:
    """Refuse, by product name, when a requested product cannot be drawn.

    Group keywords (``all``, ``derived``, ...) expand to whatever IS
    renderable, so they never refuse; an explicitly named product that the
    catalog marks non-renderable stops the run before anything is drawn,
    so a partial gallery cannot pass as the requested one."""

    requested = [token.strip() for token in products.split(",")
                 if token.strip()]
    for token in requested:
        if token in _GROUP_KEYWORDS:
            continue
        row = rows.get(token)
        if row is None:
            known = ", ".join(sorted(slug for slug, (_k, status, _d)
                                     in rows.items()
                                     if status == "renderable"))
            raise RenderDoorError(
                f"product '{token}' is not in the renderer catalog for "
                f"{history}; nothing by that name can be drawn. Renderable "
                f"products from this history: {known}."
            )
        kind, status, detail = row
        if status != "renderable":
            hint = (f" (fields absent from the source history: "
                    f"{', '.join(absent)})" if absent else "")
            raise RenderDoorError(
                f"product '{token}' is not renderable from {history}: "
                f"{status} -- {detail}{hint}. The render stops before "
                f"drawing anything so a partial gallery cannot pass as the "
                f"requested one; drop the product from --products or rerun "
                f"the model with the fields it needs."
            )


def render_frame(renderer_exe: Path, wrfout: Path, *, scratch: Path,
                 index: int, products: str, frames: str, width: int,
                 height: int, heavy: bool) -> tuple[list[tuple[str, Path]],
                                                    list[dict[str, str]],
                                                    list[str], list[str]]:
    """One renderer invocation, one per-frame store (campaign convention).

    Returns ``(rendered, skipped, failed, invocation)`` where ``rendered``
    is ``[(slug, png_path), ...]``.  Only FAILED is a failure; an honest
    SKIPPED is relayed, never hidden."""

    store = Path(scratch) / f"store-{index:03d}"
    raw_out = Path(scratch) / f"raw-{index:03d}"
    store.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)
    command = [
        str(renderer_exe),
        "--store-root", str(store),
        "--out-dir", str(raw_out),
        "--products", products,
        "--frames", frames,
        "--width", str(width),
        "--height", str(height),
    ]
    if heavy:
        command.append("--heavy")
    command.append(str(wrfout))
    result = _run(command, log_prefix="RENDER")
    rendered: list[tuple[str, Path]] = []
    skipped: list[dict[str, str]] = []
    failed: list[str] = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("RENDERED "):
            slug, _, path = line[len("RENDERED "):].partition(" ")
            if path:
                rendered.append((slug, Path(path)))
        elif line.startswith("SKIPPED "):
            slug, _, reason = line[len("SKIPPED "):].partition(" ")
            skipped.append({"product": slug,
                            "reason": reason or "no reason given"})
    for line in (result.stderr or "").splitlines():
        if line.startswith("FAILED "):
            failed.append(line[len("FAILED "):])
    if result.returncode != 0 and not failed:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        failed.append(tail[-1] if tail else f"exit {result.returncode}")
    return rendered, skipped, failed, [str(part) for part in command]


def run_render(args) -> int:
    """The ``mpas-port render`` handler.  Returns the process exit code."""

    started = time.monotonic()
    out_root = Path(args.out)
    scratch = Path(args.scratch) if args.scratch else default_scratch(out_root)
    refuse_scratch_inside_out(scratch, out_root)

    # Argument coherence FIRST, before any binary is looked for: an
    # incoherent command is incoherent whether or not an engine is
    # installed, and making the user install one to be told so wastes
    # their time and hides the real complaint behind a second one.
    compose = getattr(args, "compose", None)
    compose_base_only = bool(getattr(args, "compose_base_only", False))
    if compose is not None:
        if args.history or args.mesh:
            raise RenderDoorError(
                "--compose owns the mesh and history list; passing --mesh "
                "or --history as well would leave two source lists that "
                "can disagree about which run is the base. Put every run "
                "in the compose file."
            )
    else:
        if compose_base_only:
            raise RenderDoorError(
                "--compose-base-only describes what to leave OUT of a "
                "composite, and there is no composite without --compose. "
                "A single-source render is already the base alone."
            )
        if args.window == "composite":
            raise RenderDoorError(
                "--window composite is sized to the fine regions named in "
                "--compose, and no source list was given, so there is no "
                "fine region to frame. Use --window mesh for one run's own "
                "refined core, or add --compose."
            )
        if not args.mesh or not args.history:
            raise RenderDoorError(
                "--history and --mesh are required unless --compose names "
                "them instead."
            )

    convert_exe = resolve_convert_exe(args.convert_exe)
    renderer_exe = resolve_renderer_exe(args.renderer_exe)
    if compose is not None:
        compose = _require_file(
            Path(compose), "--compose",
            "Give a JSON file with a 'base' entry and an 'overlays' list, "
            "each carrying 'label', 'mesh' and 'history'.")
        mesh, histories = None, None
    else:
        mesh = _require_file(
            Path(args.mesh), "--mesh",
            "The converter cannot place Voronoi cells on the render grid "
            "without the mesh's cell coordinates; point --mesh at the run's "
            "grid/static netCDF (the file the run was initialized on).")
        histories = [
            _require_file(
                Path(item), "--history",
                "Give the run's history.*.nc output files; diag files do "
                "not carry the 3-D state the converter reads.")
            for item in args.history
        ]

    try:
        width_text, height_text = str(args.size).lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError:
        raise RenderDoorError(
            f"--size must be WIDTHxHEIGHT (e.g. 1200x900), got {args.size!r}"
        ) from None

    scratch.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    invocations: list[list[str]] = []

    report, convert_invocation = convert_frames(
        convert_exe, histories=histories, mesh=mesh, scratch=scratch,
        window=args.window, field_set=args.field_set,
        simulation_start=args.simulation_start, nc_format=args.nc_format,
        compose=compose, compose_base_only=compose_base_only)
    invocations.append(convert_invocation)

    frames_meta = report.get("frames", [])
    if compose is not None:
        # A composite converts only the hours EVERY source carries, so its
        # frame count is an intersection and cannot be checked against one
        # history list.  What still has to hold is that it produced
        # something: an empty series would otherwise reach the renderer as
        # a clean run that delivered no images.
        if not frames_meta:
            raise RenderDoorError(
                f"rw_mpas_convert composed no frames from {compose}; no "
                f"valid time is carried by every source in the list, so "
                f"there is nothing to render."
            )
        histories = [Path(meta["output"]) for meta in frames_meta]
    elif len(frames_meta) != len(histories):
        raise RenderDoorError(
            f"rw_mpas_convert reported {len(frames_meta)} frames for "
            f"{len(histories)} history files; the receipt does not cover "
            f"the request, so the render does not proceed."
        )

    # Coverage gate: one real catalog per distinct absent-field signature.
    # Frames from one converter run differ only when the histories
    # themselves carry different fields.
    catalogs: dict[tuple[str, ...], dict] = {}
    catalog_summary = ""
    for meta, history in zip(frames_meta, histories):
        signature = tuple(meta.get("absent_wrf_fields", []))
        if signature not in catalogs:
            rows, catalog_summary, catalog_invocation = list_catalog(
                renderer_exe, Path(meta["output"]), scratch=scratch,
                heavy=args.heavy)
            catalogs[signature] = rows
            invocations.append(catalog_invocation)
        refuse_uncovered_products(
            args.products, catalogs[signature], history,
            list(meta.get("absent_wrf_fields", [])))

    results: list[FrameResult] = []
    total_rendered = total_skipped = total_failed = 0
    for index, (meta, history) in enumerate(zip(frames_meta, histories)):
        frame = FrameResult(
            history=str(history),
            wrfout=str(meta["output"]),
            valid_time=str(meta.get("valid_time", "")),
            valid_day=valid_day(meta.get("valid_time")),
            output_sha256=str(meta.get("output_sha256", "")),
            absent_wrf_fields=list(meta.get("absent_wrf_fields", [])),
            seam=list(meta.get("seam", [])),
        )
        rendered, skipped, failed, invocation = render_frame(
            renderer_exe, Path(meta["output"]), scratch=scratch, index=index,
            products=args.products, frames=args.frames, width=width,
            height=height, heavy=args.heavy)
        invocations.append(invocation)
        frame.skipped = skipped
        frame.failed = failed
        for slug, png in rendered:
            domain, product = split_engine_name(png.name, slug)
            destination = place_path(out_root, domain, product,
                                     frame.valid_day, png.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(png), str(destination))
            frame.rendered.append(slug)
            frame.placed.append(
                str(destination.relative_to(out_root)).replace(os.sep, "/"))
        results.append(frame)
        total_rendered += len(frame.rendered)
        total_skipped += len(skipped)
        total_failed += len(failed)
        print(f"FRAME {frame.valid_time} rendered={len(frame.rendered)} "
              f"skipped={len(skipped)} failed={len(failed)}", flush=True)
        for item in skipped:
            print(f"  SKIPPED {item['product']} {item['reason']}",
                  flush=True)
        for item in failed:
            print(f"  FAILED {item}", flush=True)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "engine": {
            "convert_exe": str(convert_exe),
            "convert_exe_sha256": sha256_file(convert_exe),
            "renderer_exe": str(renderer_exe),
            "renderer_exe_sha256": sha256_file(renderer_exe),
        },
        "convert": {
            "schema": report.get("schema"),
            "engine": report.get("engine"),
            "window": report.get("window"),
            "window_spec": report.get("window_spec"),
            "mesh": str(mesh) if mesh is not None else None,
            "mesh_sha256": report.get("mesh_sha256"),
            "weights_sha256": report.get("weights_sha256"),
            "resample_method": report.get("resample_method"),
            "field_set": report.get("field_set"),
            # Present only on a composed render: which runs went in, how
            # many grid points each won, and how wide the seam is.  A
            # composite whose manifest did not say which runs it came from
            # would be indistinguishable from a single-run render.
            "compose_spec": str(compose) if compose is not None else None,
            "compose_base_only": compose_base_only,
            "composite": report.get("composite"),
        },
        "catalog_summary": catalog_summary,
        "products_requested": args.products,
        "frames_requested": args.frames,
        "size": f"{width}x{height}",
        "counts": {"rendered": total_rendered, "skipped": total_skipped,
                   "failed": total_failed},
        "frames": [vars(frame) for frame in results],
        "invocations": invocations,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    # A second run into the same delivered tree is legitimate (one full
    # catalog pass plus a restricted pass is the proof shape); its receipt
    # must not overwrite the first run's.
    manifest_path = out_root / "render-manifest.json"
    if manifest_path.exists():
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest_path = out_root / f"render-manifest-{stamp}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"MANIFEST {manifest_path}", flush=True)

    ok = total_failed == 0 and total_rendered > 0
    if ok and not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"SCRATCH cleared {scratch}", flush=True)
    else:
        print(f"SCRATCH kept {scratch}", flush=True)
    print(f"DOOR rendered={total_rendered} skipped={total_skipped} "
          f"failed={total_failed} out={out_root}", flush=True)
    if total_rendered == 0:
        print("DOOR nothing rendered: every requested product was skipped "
              "or failed; see the manifest's frames[] for each reason",
              flush=True)
        return 1
    return 0 if total_failed == 0 else 1


def add_render_arguments(parser) -> None:
    """Wire the ``render`` subcommand's arguments onto ``parser``."""

    parser.add_argument(
        "--history", nargs="+", default=None, metavar="FILE",
        help="MPAS history netCDF file(s), one frame each. Required unless "
             "--compose names them instead")
    parser.add_argument(
        "--mesh", default=None, metavar="FILE",
        help="the run's grid/static netCDF with cell coordinates. Required "
             "unless --compose names it instead")
    parser.add_argument(
        "--compose", default=None, metavar="FILE",
        help="compose SEVERAL runs of the same hours into one frame: the "
             "coarse run everywhere, each fine run inside the ground that "
             "fine run actually resolves. The file lists a base and any "
             "number of overlays, each with its own mesh and history "
             "files. Pair it with --window composite, which frames all the "
             "fine regions at once. THE BREAKAGE THIS PREVENTS, MEASURED "
             "(2026-08-26): a cascade produces a coarse global forecast and "
             "one fine forecast per placed grid, and the only way to look "
             "at both was two separate images at two different scales. "
             "Compositing them as PICTURES would need every pixel's "
             "geographic transform and would still leave two projections "
             "and two colour scales to reconcile; compositing the DATA is "
             "one resample onto one grid and one render, so no seam can "
             "come from the drawing")
    parser.add_argument(
        "--compose-base-only", action="store_true",
        help="with --compose, frame the same window from the same sources "
             "but convert the BASE alone. The coarse counterpart to a "
             "composite, over identical ground, so a comparison between "
             "them cannot be confounded by a different map")
    parser.add_argument(
        "--out", required=True, metavar="DIR",
        help="delivered PNG tree root (<domain>/<product>/<valid-day>/)")
    parser.add_argument(
        "--products", default="all",
        help="renderer product selection: all|direct|derived|heavy|windowed "
             "or comma-separated slugs (default: all)")
    parser.add_argument(
        "--frames", default="all",
        help="renderer frame selection within each store (default: all)")
    parser.add_argument(
        "--window", default="mesh",
        choices=("mesh", "focus", "global", "composite"),
        help="converter render window. 'mesh' is DERIVED from the mesh you "
             "pass: the cells within 2x the minimum spacing are its refined "
             "region, and the window is centred on them at their median "
             "spacing. 'focus' is a fixed 22 km Lambert box over the CONUS "
             "and 'global' a fixed 0.25 degree overview; both are frozen "
             "geometry whose weights digests are recorded in evidence. "
             "THE BREAKAGE 'mesh' PREVENTS, MEASURED (2026-08-26): a mesh "
             "this project PLACED had its refined 4.5 km core in the "
             "tropical Atlantic, and the only windows that existed were a "
             "CONUS box 5,000 km away -- which samples nothing but 75 km "
             "background cells -- and a 0.25 degree (about 28 km) global "
             "grid, which cannot resolve a 4.5 km core at all. There was no "
             "way to render the run's own fine region, and every future "
             "placement would have needed another hardcoded box. "
             "'composite' needs --compose and frames every fine region in "
             "the source list at once (default: mesh)")
    parser.add_argument(
        "--field-set", default="full", choices=("full", "surface"),
        help="converter field set (default: full)")
    parser.add_argument(
        "--simulation-start", default=None, metavar="YYYY-MM-DD_HH:MM:SS",
        help="run start, for true lead hours in filenames")
    parser.add_argument(
        "--size", default="1200x900",
        help="output pixels as WIDTHxHEIGHT (default: 1200x900)")
    parser.add_argument(
        "--heavy", action="store_true",
        help="include the heavy product family (off by default)")
    parser.add_argument(
        "--nc-format", default="cdf2", choices=("cdf2", "cdf5"),
        help="converted frame netCDF variant (default: cdf2)")
    parser.add_argument(
        "--scratch", default=None, metavar="DIR",
        help="conversion/render scratch dir OUTSIDE --out "
             "(default: sibling <out>.render-scratch)")
    parser.add_argument(
        "--keep-scratch", action="store_true",
        help="keep the scratch dir after a clean run")
    parser.add_argument(
        "--convert-exe", default=None, metavar="FILE",
        help=f"rw_mpas_convert binary (else ${CONVERT_ENV}, then PATH)")
    parser.add_argument(
        "--renderer-exe", default=None, metavar="FILE",
        help=f"rw_wrfbatch binary (else ${RENDERER_ENV}, "
             f"${RENDERER_ENV_GPUWM}, then PATH)")
