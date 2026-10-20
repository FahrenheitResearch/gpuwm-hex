"""Prepare one native step with real initialization and controlled increments.

Metadata orchestration only; increment_fixture.rs writes all field data.
This is column evidence and cannot replace the required IAU forecast experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--suite", required=True,
                        choices=("none", "mesoscale_reference", "convection_permitting"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--dt", type=int, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    case, native = args.case.resolve(), args.native.resolve()
    init, = case.glob("*.init.nc")
    mesh = init.name.removesuffix(".init.nc")
    for source in [*case.glob("*.nc"), *case.glob("*.graph.info*")]:
        (run/source.name).symlink_to(source)
    for source in native.iterdir():
        if source.is_file() and (source.suffix in (".TBL", ".DBL") or
                source.name.startswith(("stream_list.", "OZONE_")) or
                source.name in ("CCN_ACTIVATE_DATA", "RRTMG_LW_DATA", "RRTMG_SW_DATA")):
            (run/source.name).symlink_to(source)
    (run/"atmosphere_model").symlink_to(args.oracle.resolve())
    if args.tables:
        for source in args.tables.resolve().glob("*.DBL"):
            if not (run/source.name).exists():
                (run/source.name).symlink_to(source)
    subprocess.run([str(args.fixture.resolve()), str(init), str(run/"increments.nc")], check=True)
    seconds = args.dt
    duration = f"{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}"
    namelist = f"""&nhyd_model
 config_start_time = '{args.start}'
 config_run_duration = '{duration}'
 config_dt = {args.dt}
 config_time_integration_order = 2
 config_number_of_sub_steps = 4
 config_dynamics_split_steps = 3
/
&decomposition
 config_block_decomp_file_prefix = '{mesh}.graph.info.part.'
/
&io
 config_pio_num_iotasks = 0
 config_pio_stride = 1
/
&physics
 config_sst_update = false
 config_physics_suite = '{args.suite}'
/
&iau
 config_IAU_option = 'on'
 config_IAU_window_length_s = 21600.0
/
"""
    (run/"namelist.atmosphere").write_text(namelist)
    (run/"streams.atmosphere").write_text(f"""<streams>
<immutable_stream name="input" type="input" filename_template="{init.name}" input_interval="initial_only" />
<immutable_stream name="iau" type="input" filename_template="increments.nc" filename_interval="none" input_interval="initial_only" packages="iau" />
<immutable_stream name="restart" type="input;output" filename_template="restart.nc" input_interval="initial_only" output_interval="none" />
<stream name="output" type="output" filename_template="output.nc" output_interval="{duration}"><file name="stream_list.atmosphere.output" /></stream>
<stream name="diagnostics" type="output" filename_template="diagnostics.nc" output_interval="{duration}"><file name="stream_list.atmosphere.diagnostics" /></stream>
<stream name="surface" type="input;output" filename_template="surface.nc" input_interval="none" output_interval="{duration}"><file name="stream_list.atmosphere.surface" /></stream>
<immutable_stream name="ugwp_oro_data_in" type="input" filename_template="{mesh}.ugwp_oro_data.nc" packages="ugwp_orog_stream" input_interval="initial_only" />
</streams>
""")
    (run/"capture-provenance.json").write_text(json.dumps({
        "case_publication": str(case/"PUBLICATION.json"), "suite": args.suite,
        "native_build": str(native), "oracle": str(args.oracle.resolve()),
        "case_publication_sha256": hashlib.sha256((case/"PUBLICATION.json").read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        "fixture_source_sha256": hashlib.sha256((Path(__file__).parent/"increment_fixture.rs").read_bytes()).hexdigest(),
        "increment_origin": "controlled constants, not analysis minus native forecast",
        "process_count": 1, "native_execution_host": os.uname().nodename,
        "start": args.start, "dt": args.dt, "window": 21600.0,
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
