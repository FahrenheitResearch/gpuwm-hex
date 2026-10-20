"""Prepare a real published native executable probe; no weather decoding.

Inputs and executable remain read-only. This probe exercises DD, its output
switch and the TKE budget on f0's published host-acceptance initial condition.
It is a native-defect measurement, not a lane forecast acceptance deck.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

FIELDS = ("qke", "qke_adv", "tke_pbl", "el_pbl", "dqke", "qbuoy", "qdiss",
          "qshear", "qwt", "cov", "qsq", "tsq", "qc_bl", "qi_bl",
          "cldfrac_bl", "edmf_a", "edmf_w", "edmf_qt", "edmf_qc",
          "edmf_thl", "edmf_ent", "hpbl", "kpbl", "kzh", "kzm", "kzq")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def link_readonly(path: Path, target: Path):
    target = target.resolve()
    if path.is_symlink() and path.resolve() == target:
        return
    path.symlink_to(target)


def prepare(work: Path, native: Path, input_file: Path, template: Path, registry: Path,
            mesh: str, mesh_root: Path, dt: int, duration: str):
    work.mkdir(parents=True, exist_ok=True)
    namelist = template.read_text()
    namelist = re.sub(r"(config_run_duration\s*=\s*)['\"][^'\"]+['\"]", r'\g<1>"'+duration+'"', namelist)
    namelist = re.sub(r"(config_dt\s*=\s*)[\d.]+", r'\g<1>'+str(dt), namelist)
    namelist = re.sub(r"(config_block_decomp_file_prefix\s*=\s*)['\"][^'\"]+['\"]",
                      r'\g<1>"'+mesh+'.graph.info.part."', namelist)
    # The template may already select the forced companion. Set each option
    # once, preserving every unrelated native deck option.
    for name in ("config_pbl_scheme", "config_sfclayer_scheme", "config_mynn_edmf_dd",
                 "config_mynn_edmf_output", "config_mynn_tkebudget"):
        namelist = re.sub(r"^\s*"+name+r"\s*=.*$", "", namelist, flags=re.M)
    namelist = re.sub(r"config_physics_suite\s*=\s*['\"][^'\"]+['\"]", lambda match: match[0] + '''
    config_pbl_scheme = "bl_mynn"
    config_sfclayer_scheme = "sf_mynn"
    config_mynn_edmf_dd = 1
    config_mynn_edmf_output = 1
    config_mynn_tkebudget = 1''', namelist)
    (work / "namelist.atmosphere").write_text(namelist)
    link_readonly(work / (mesh+".init.nc"), input_file)
    for filename in native.iterdir():
        if filename.name not in ("namelist.atmosphere", "streams.atmosphere") and filename.is_file():
            link_readonly(work / filename.name, filename)
    for name in (mesh+".graph.info", mesh+".graph.info.part.4"):
        link_readonly(work / name, mesh_root / name)
    streams = ET.Element("streams")
    ET.SubElement(streams, "immutable_stream", name="input", type="input",
                  filename_template=mesh+".init.nc", input_interval="initial_only")
    ET.SubElement(streams, "immutable_stream", name="restart", type="input;output",
                  filename_template="restart.$Y-$M-$D_$h.$m.$s.nc",
                  input_interval="initial_only", output_interval="none")
    output = ET.SubElement(streams, "stream", name="output", type="output",
                           filename_template="mynn.$Y-$M-$D_$h.$m.$s.nc", output_interval="0_00:06:00")
    for name in FIELDS:
        ET.SubElement(output, "var", name=name)
    diagnostics = ET.SubElement(streams, "stream", name="diagnostics", type="output",
        filename_template="diagnostics.$Y-$M-$D_$h.$m.$s.nc", output_interval="0_00:06:00")
    ET.SubElement(diagnostics, "var", name="hpbl")
    surface = ET.SubElement(streams, "stream", name="surface", type="input",
                            filename_template="unused.sfc.nc", filename_interval="none", input_interval="none")
    ET.SubElement(surface, "file", name="stream_list.atmosphere.surface")
    ET.SubElement(streams, "immutable_stream", name="iau", type="input", packages="iau",
                  filename_template="unused.iau.nc", input_interval="initial_only")
    ET.SubElement(streams, "immutable_stream", name="lbc_in", type="input", packages="limited_area",
                  filename_template="unused.lbc.nc", input_interval="3:00:00")
    ET.indent(streams)
    ET.ElementTree(streams).write(work / "streams.atmosphere", encoding="unicode")
    defined = {var.attrib["name"] for var in ET.parse(registry).iter("var")}
    assert set(FIELDS) <= defined
    metadata = {"scope": "real native DD measurement, not hex acceptance",
                "native_executable": str(native / "atmosphere_model"),
                "native_sha256": sha(native / "atmosphere_model"),
                "input": str(input_file), "input_sha256": sha(input_file),
                "namelist_sha256": sha(work / "namelist.atmosphere"),
                "streams_sha256": sha(work / "streams.atmosphere"),
                "fields": FIELDS, "dt": dt, "duration": duration, "excluded_words": [],
                "absent_from_registry": [name for name in
                    ("edmf_a_dd", "edmf_w_dd", "edmf_qt_dd", "edmf_thl_dd", "edmf_ent_dd", "edmf_qc_dd")
                    if name not in defined]}
    (work / "probe.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "native", "input", "template", "registry"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--dt", type=int, required=True)
    parser.add_argument("--duration", required=True)
    args = parser.parse_args()
    prepare(args.work, args.native, args.input, args.template, args.registry,
            args.mesh, args.mesh_root, args.dt, args.duration)
