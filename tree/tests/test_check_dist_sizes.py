"""The upload cap is exclusive, including at its exact byte boundary."""
import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location("check_dist_sizes", Path(__file__).resolve().parents[1]
                                            / "tools/check_dist_sizes.py")
sizes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sizes)


@pytest.mark.parametrize("size,status,code", [
    (94_999_999, "PASS", 0), (95_000_000, "WARN", 0), (99_999_999, "WARN", 0),
    (100_000_000, "FAIL", 1), (100_000_001, "FAIL", 1),
])
def test_exact_cap_and_warning_boundaries(tmp_path, capsys, size, status, code):
    path = tmp_path / "boundary.tar"
    with tarfile.open(path, "w") as archive:
        row = tarfile.TarInfo("member")
        row.size = 1
        archive.addfile(row, io.BytesIO(b"x"))
    with path.open("r+b") as stream:
        stream.truncate(size)  # sparse padding, with a valid tar end marker
    assert sizes.main([str(path)]) == code
    output = capsys.readouterr().out
    assert f"{status} {path}: {size} B; margin {100_000_000 - size} B" in output
    assert ("WARNING:" in output) == (size >= 95_000_000)
    assert "member" in output


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_ten_largest_members_are_sorted(tmp_path, kind):
    path = tmp_path / ("test.whl" if kind == "wheel" else "test.tar.gz")
    if kind == "wheel":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("directory/", b"")
            for size in range(12):
                archive.writestr(f"member-{size}", b"x" * size)
    else:
        with tarfile.open(path, "w:gz") as archive:
            for size in range(12):
                row = tarfile.TarInfo(f"member-{size}")
                row.size = size
                archive.addfile(row, io.BytesIO(b"x" * size))
    assert sizes.largest_members(path) == [(size, f"member-{size}") for size in range(11, 1, -1)]


def test_bad_and_missing_archives_fail(tmp_path, capsys):
    path = tmp_path / "broken.whl"
    path.write_bytes(b"not a zip")
    assert sizes.main([str(path), str(tmp_path / "missing.tar.gz")]) == 1
    assert capsys.readouterr().out.count("FAIL") == 2
