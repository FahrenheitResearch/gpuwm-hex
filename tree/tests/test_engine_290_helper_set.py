"""The release engine binds all six helpers, without relaxing mixed-set refusal."""
import json
from pathlib import Path

from hexcore import cuda_arwen_helper_scope_v2 as scope


def test_290_source_set_uses_ordinary_route_and_declares_numerics_change():
    mode = "engine-2.9.0-source"
    assert set(scope.ADMITTED_SOURCE_SETS[mode]) == set(scope.PUBLISHED_SOURCES)
    assert mode in scope.PUBLISHED_MODES
    assert "source identity only" in scope.RUNTIME_DESCRIPTIONS[mode]
    rows = json.loads(Path(scope.__file__).with_name("data").joinpath(
        "profiles", "engine-helper-sets.v1.json").read_text())["sets"]
    row = next(row for row in rows if row["id"] == mode)
    assert row["engine_commit"] == "0bf5057645f571d737e723fda3ab6c1de9b3c69d"
    assert all(len(digest) == 64 for digest in row["sources"].values())
    changed = {name for name, digest in row["sources"].items()
               if digest != scope.ADMITTED_SOURCE_SETS["engine-2.8.9-source"][name]}
    assert changed == {"gpuwm/core/rrtmg_sw.py", "gpuwm/core/rrtmg_legacy_prep.py"}
    assert "changes forecast numerics" in row["description"]
