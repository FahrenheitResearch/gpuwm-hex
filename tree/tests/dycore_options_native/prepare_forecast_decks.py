"""Prepare native option proof decks; execution requires the integrated door.

Registry.xml is the namelist grouping authority. This changes test inputs only,
never application defaults. The caller supplies the matrix and published base.
Dry/moist input-artifact preparation belongs to the forecast verification tool.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from hexcore.config import parse_fortran_namelist


def fortran_scalar(value):
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def prepare(matrix_path, templates, registry_path, output, receipt):
    matrix = json.loads(matrix_path.read_text())
    registry = ET.parse(registry_path).getroot()
    groups = {option.attrib["name"]: record.attrib["name"]
              for record in registry.iter("nml_record")
              for option in record.iter("nml_option")}
    if output.exists():
        raise ValueError("use a fresh forecast deck directory")
    rows = []
    for index, deck in enumerate(matrix["decks"]):
        date = datetime.fromisoformat(deck["date"]).strftime("%Y-%m-%d_%H.%M.%S")
        base = templates / deck["mesh"] / date / "suite-none"
        namelist = {group: dict(values) for group, values in
                    parse_fortran_namelist((base / "namelist.atmosphere").read_text()).items()}
        if namelist["physics"]["config_physics_suite"] != "none":
            raise ValueError("the native baseline must have physics off")
        # Make the native selector and nonpositive nominalMinDc fallback
        # explicit; do not inherit the application's mixing defaults.
        # Registry.xml:99 defaults split transport to true. Spell it out
        # because the existing application type retains its own defaults.
        overrides = {"config_split_dynamics_transport": True,
                     "config_horiz_mixing": "2d_smagorinsky",
                     "config_len_disp": 0.0, **deck["parameters"]}
        for name, value in overrides.items():
            namelist.setdefault(groups[name], {})[name] = value
        target = output / f"deck-{index:03d}"
        target.mkdir(parents=True)
        text = "".join("&" + group + "\n" + "".join(
            f" {name} = {fortran_scalar(value)}\n" for name, value in values.items())
            + "/\n" for group, values in namelist.items())
        parsed = parse_fortran_namelist(text)
        assert all(parsed[groups[name]][name] == value for name, value in overrides.items())
        (target / "namelist.atmosphere").write_text(text)
        streams = (base / "streams.atmosphere").read_bytes()
        (target / "streams.atmosphere").write_bytes(streams)
        sha = lambda data: hashlib.sha256(data).hexdigest()
        rows.append({"deck_index": index, "matrix_entry": deck,
                     "base_directory": str(base), "directory": str(target),
                     "base_namelist_sha256": sha((base / "namelist.atmosphere").read_bytes()),
                     "namelist_sha256": sha(text.encode()), "streams_sha256": sha(streams),
                     "required_input_state": deck["state"],
                     "status": "PREPARED; input state and forecast execution pending"})
    receipt.write_text(json.dumps({"scope": "native namelist/stream orchestration only; no forecast claim",
                                  "decks_prepared": len(rows), "decks": rows}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("matrix", "templates", "registry", "output", "receipt"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.matrix, args.templates, args.registry, args.output, args.receipt)
