#!/usr/bin/env python3
"""Mint and verify a model-timestep anchor for the frozen v8.4.1 lane.

``hexcore.dt_admission`` refuses any ``config_dt`` holding no anchor.  This
is the runnable procedure that mints one, and the verifier that decides
whether a registered row is telling the truth.

An anchor has two halves and they cost very different things.

**The schedule receipt (host only, seconds).**  Everything a timestep can
get wrong before it ever reaches a card: the physics cadence step counts,
the Grell-Freitas ``cudt == dt`` law, the RK schedule's shape against the
proven 120 s schedule, the WSM6 minor-loop split, and clock closure over the
run length.  ``--dt`` mints it; ``--verify`` re-derives every registered
row's receipt and compares it to the JSON the row names, so a row cannot
point at a file that says something else.

**The integration anchor (one card, hours).**  Two forecasts of the same
case at this timestep whose history is byte-identical under masked digests,
finite at every step.  ``--integration-plan`` prints the exact commands and
the measured cost; this tool does not run them, because a forecast is the
forecast door's job and duplicating it here would create a second execution
path for the same thing.

**The mutation control.**  ``--mutation-control`` proves the verifier has
teeth in BOTH directions: it must pass the real 120 s row and it must fail
every fabricated variant of it -- a receipt file that does not exist, a
receipt whose dt does not match the row, a receipt whose cadence counts were
edited, and a row whose declared cadences contradict its own receipt.  A
verifier that only ever passes is not evidence of anything.

Nothing here registers anything.  Registering a second anchor moves the
frozen lane off its proven timestep and that is a ruling, not a tool run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexcore import dt_admission  # noqa: E402
from hexcore.dt_admission import DtAdmissionError, DtAnchor  # noqa: E402

SCHEMA = "gpuwm-hex.dt-anchor-mint/v1"

#: The archived 120 s constants this harness must reproduce exactly, read
#: from where the frozen proofs already record them.  This is the harness's
#: KNOWN-ANSWER arm: an instrument that cannot reproduce the one timestep
#: this project has actually proven is not an instrument.
ARCHIVED_120S = {
    "scalar_stage_timesteps": (40.0, 60.0, 120.0),
    "dynamics_stage_timesteps": (40.0 / 3.0, 20.0, 40.0),
    "dynamics_stage_acoustic_steps": (1, 3, 6),
    "stepra": 5,
    "stepbl": 1,
    "stepcu": 1,
    "minor_loops": 1,
    "source": (
        "tools/run_cuda_v841_real_x4.py::EXPECTED_SCALAR_STAGE_TIMESTEPS / "
        "EXPECTED_DYNAMICS_STAGE_TIMESTEPS / EXPECTED_STAGE_ACOUSTIC_STEPS, "
        "and tools/compare_v841_compiled_endpoint.py's dynamics/scalar stage "
        "tables"
    ),
}


class MintError(RuntimeError):
    """The harness refuses to mint or to certify."""


#: Top-level directories an anchor may cite a path inside.  A citation that
#: does not start with one of these is prose (a native-run description, a
#: ``module::SYMBOL`` reference) and is deliberately NOT checked for
#: existence -- checking it would turn a sentence into a missing file.
_EVIDENCE_ROOTS = ("evidence/", "tools/", "src/", "docs/", "verification/", "oracle/")


def _repository_paths(citation: str) -> list[str]:
    """The repository paths inside one comma-separated evidence citation."""

    paths: list[str] = []
    for part in citation.split(","):
        entry = part.strip().rstrip(".").split(" ")[0].split("::")[0]
        if entry.startswith(_EVIDENCE_ROOTS):
            paths.append(entry)
    return paths


def mint_schedule_receipt(
    dt_seconds: float,
    *,
    radiation_seconds: float = dt_admission.RADIATION_CADENCE_SECONDS,
    surface_pbl_seconds: float | None = None,
    cumulus_seconds: float | None = None,
    cumulus_scheme: str | None = "gf",
    run_steps: int = 720,
) -> dict[str, Any]:
    """Mint one timestep's schedule receipt, with the known-answer arm run."""

    receipt = dt_admission.schedule_receipt(
        dt_seconds,
        radiation_seconds=radiation_seconds,
        surface_pbl_seconds=surface_pbl_seconds,
        cumulus_seconds=cumulus_seconds,
        cumulus_scheme=cumulus_scheme,
        run_steps=run_steps,
    )
    receipt["schema_mint"] = SCHEMA
    receipt["known_answer_control"] = known_answer_control()
    return receipt


def known_answer_control() -> dict[str, Any]:
    """Re-derive the archived 120 s constants and compare them exactly.

    Runs on every mint, not only on ``--dt 120``: the instrument that
    measures an unproven timestep is the same instrument, so it declares its
    calibration in the receipt of every timestep it mints.
    """

    receipt = dt_admission.schedule_receipt(dt_admission.PROVEN_DT_SECONDS)
    checks = {
        "scalar_stage_timesteps": (
            tuple(receipt["rk_schedule"]["scalar_stage_timesteps"])
            == ARCHIVED_120S["scalar_stage_timesteps"]
        ),
        "dynamics_stage_timesteps": (
            tuple(receipt["rk_schedule"]["dynamics_stage_timesteps"])
            == ARCHIVED_120S["dynamics_stage_timesteps"]
        ),
        "dynamics_stage_acoustic_steps": (
            tuple(receipt["rk_schedule"]["dynamics_stage_acoustic_steps"])
            == ARCHIVED_120S["dynamics_stage_acoustic_steps"]
        ),
        "stepra": receipt["cadences"]["stepra"] == ARCHIVED_120S["stepra"],
        "stepbl": receipt["cadences"]["stepbl"] == ARCHIVED_120S["stepbl"],
        "stepcu": receipt["cadences"]["stepcu"] == ARCHIVED_120S["stepcu"],
        "minor_loops": receipt["wsm6"]["minor_loops"] == ARCHIVED_120S["minor_loops"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise MintError(
            "the schedule instrument does not reproduce the archived 120 s "
            f"constants: {failed} disagree with {ARCHIVED_120S['source']}.  "
            "Nothing it says about an unproven timestep can be trusted"
        )
    return {
        "reproduces_archived_120s": True,
        "checks": checks,
        "source": ARCHIVED_120S["source"],
    }


def verify_anchor(anchor: DtAnchor, *, root: Path = ROOT) -> dict[str, Any]:
    """Certify one registered anchor against the evidence it names.

    Refuses, by name, when: a named evidence path is absent; the schedule
    receipt on disk is not valid JSON; its timestep is not the row's; its
    cadences are not the row's; or re-deriving the receipt from the row's own
    declarations does not reproduce the stored numbers.
    """

    findings: list[str] = []
    if anchor.admitted_on.startswith("CANDIDATE"):
        return {
            "dt_seconds": anchor.dt_seconds,
            "certified": False,
            "findings": [
                f"dt={anchor.dt_seconds:g} s is a CANDIDATE row admitted only "
                f"for the duration of a mint run ({anchor.basis}).  A "
                f"candidate is never an anchor: it names no schedule receipt "
                f"on disk and no integration pair, and certifying one would "
                f"be certifying the run that is still trying to earn it"
            ],
        }
    receipt_path = root / anchor.schedule_receipt
    if not receipt_path.exists():
        findings.append(
            f"dt={anchor.dt_seconds:g} s names a schedule receipt that is not "
            f"in the tree: {anchor.schedule_receipt}"
        )
        stored: dict[str, Any] | None = None
    elif receipt_path.is_dir():
        findings.append(
            f"dt={anchor.dt_seconds:g} s names a DIRECTORY as its schedule "
            f"receipt: {anchor.schedule_receipt}; an anchor names the file "
            f"whose numbers can be compared, not the folder around it"
        )
        stored = None
    else:
        try:
            stored = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            findings.append(
                f"dt={anchor.dt_seconds:g} s names a schedule receipt that "
                f"does not read as JSON ({error})"
            )
            stored = None

    if stored is not None:
        if float(stored.get("dt_seconds", float("nan"))) != float(anchor.dt_seconds):
            findings.append(
                f"dt={anchor.dt_seconds:g} s names a schedule receipt minted "
                f"at dt={stored.get('dt_seconds')!r}: the row and its evidence "
                f"are about different timesteps"
            )
        cadences = stored.get("cadences") or {}
        for field, declared in (
            ("radiation_seconds", anchor.radiation_seconds),
            ("surface_pbl_seconds", anchor.surface_pbl_seconds),
            ("cumulus_seconds", anchor.cumulus_seconds),
        ):
            recorded = cadences.get(field)
            same = (recorded is None and declared is None) or (
                recorded is not None
                and declared is not None
                and float(recorded) == float(declared)
            )
            if not same:
                findings.append(
                    f"dt={anchor.dt_seconds:g} s declares {field}={declared!r} "
                    f"but its schedule receipt records {recorded!r}"
                )
        try:
            rederived = dt_admission.schedule_receipt(
                anchor.dt_seconds,
                radiation_seconds=anchor.radiation_seconds,
                surface_pbl_seconds=anchor.surface_pbl_seconds,
                cumulus_seconds=anchor.cumulus_seconds,
                cumulus_scheme=anchor.cumulus_scheme,
                run_steps=int(
                    (stored.get("clock_closure") or {}).get("run_steps", 720)
                ),
            )
        except DtAdmissionError as error:
            findings.append(
                f"dt={anchor.dt_seconds:g} s cannot be re-derived at its own "
                f"declared cadences: {error}"
            )
        else:
            for section in ("cadences", "rk_schedule", "wsm6", "clock_closure"):
                if stored.get(section) != rederived[section]:
                    findings.append(
                        f"dt={anchor.dt_seconds:g} s: the stored {section!r} "
                        f"block does not reproduce -- the receipt on disk is "
                        f"not what this timestep actually derives"
                    )

    for label, value in (
        ("integration anchor", anchor.integration_anchor),
        ("native reference", anchor.native_reference),
    ):
        if value is None:
            continue
        for entry in _repository_paths(value):
            if not (root / entry).exists():
                findings.append(
                    f"dt={anchor.dt_seconds:g} s names a {label} path that is "
                    f"not in the tree: {entry}"
                )

    if not anchor.card or not anchor.admitted_on or not anchor.basis:
        findings.append(
            f"dt={anchor.dt_seconds:g} s is registered without a card, a date "
            f"or a basis; that is a switch, not an anchor"
        )

    return {
        "dt_seconds": anchor.dt_seconds,
        "certified": not findings,
        "findings": findings,
    }


def verify_registry(*, root: Path = ROOT) -> dict[str, Any]:
    """Certify every registered anchor."""

    results = [
        verify_anchor(anchor, root=root)
        for anchor in sorted(
            dt_admission.ADMITTED_TIMESTEPS.values(), key=lambda a: a.dt_seconds
        )
    ]
    return {
        "schema": SCHEMA,
        "registered": len(results),
        "certified": all(result["certified"] for result in results),
        "anchors": results,
    }


def _fabricate(anchor: DtAnchor, **changes: Any) -> DtAnchor:
    import dataclasses

    return dataclasses.replace(anchor, **changes)


def unanchored_probe_dt() -> float:
    """A timestep that holds no anchor and whose host half mints clean.

    DERIVED, never hardcoded.  The mutation control needs an unanchored
    timestep for its candidate arm, and any literal it named would go stale
    the moment that value was earned -- which is exactly what happened on
    2026-08-26: the control named 20.0, four anchors were minted, and the
    control's own candidate arm started refusing with "already holds an
    anchor".  A control that breaks when the thing it guards succeeds is a
    control nobody can keep.

    So it is searched: exact divisors of the radiation cadence, smallest
    first, taking the first that is unanchored and can mint a schedule
    receipt.  If every divisor in range is anchored the harness says so
    rather than fabricating one.
    """

    cadence = dt_admission.RADIATION_CADENCE_SECONDS
    for count in range(2, 4000):
        candidate = cadence / count
        if dt_admission.admitted_timestep(candidate) is not None:
            continue
        try:
            dt_admission.schedule_receipt(candidate, run_steps=1)
        except DtAdmissionError:
            continue
        return candidate
    raise MintError(
        "no unanchored timestep dividing the "
        f"{cadence:g} s radiation cadence could be found, so the mutation "
        "control has no candidate arm to run.  Either every divisor is "
        "anchored -- which would be a finding in itself -- or the schedule "
        "instrument is refusing everything"
    )


def mutation_control(*, root: Path = ROOT) -> dict[str, Any]:
    """Prove the verifier fails on fabricated evidence, and passes on real.

    Six arms.  The first must certify; the other five must not.  A verifier
    that answers "certified" to all six is measuring nothing, which is the
    failure mode this control exists to catch.
    """

    truth = dt_admission.admitted_timestep(dt_admission.PROVEN_DT_SECONDS)
    if truth is None:  # pragma: no cover - the registry lost its own anchor
        raise MintError(
            "the registry holds no anchor at the proven timestep, so the "
            "mutation control has no true arm to run"
        )

    arms: list[dict[str, Any]] = []

    def _arm(name: str, anchor: DtAnchor, expect_certified: bool) -> None:
        result = verify_anchor(anchor, root=root)
        arms.append(
            {
                "arm": name,
                "expected_certified": expect_certified,
                "certified": result["certified"],
                "agrees": result["certified"] is expect_certified,
                "findings": result["findings"],
            }
        )

    _arm("registered-truth", truth, True)
    _arm(
        "receipt-absent",
        _fabricate(truth, schedule_receipt="evidence/does-not-exist.json"),
        False,
    )
    _arm(
        "receipt-is-a-directory",
        _fabricate(truth, schedule_receipt="evidence/dt-admission-20260826"),
        False,
    )
    _arm(
        "row-claims-a-timestep-its-receipt-does-not",
        _fabricate(truth, dt_seconds=60.0, surface_pbl_seconds=60.0, cumulus_seconds=60.0),
        False,
    )
    _arm(
        "row-claims-a-cadence-its-receipt-does-not",
        _fabricate(truth, surface_pbl_seconds=600.0),
        False,
    )
    _arm(
        "integration-anchor-path-absent",
        _fabricate(truth, integration_anchor="evidence/no-such-campaign"),
        False,
    )
    probe = unanchored_probe_dt()
    with dt_admission.candidate_mint(
        probe,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="mutation control, no card",
    ) as candidate:
        _arm("candidate-row-is-never-an-anchor", candidate, False)

    disagreements = [arm["arm"] for arm in arms if not arm["agrees"]]
    return {
        "schema": SCHEMA,
        "unanchored_probe_dt_seconds": probe,
        "arms": arms,
        "has_teeth_both_directions": not disagreements,
        "disagreements": disagreements,
    }


def run_candidate_forecast(
    dt_seconds: float,
    *,
    card: str,
    authorization: str,
    forecast_argv: list[str],
) -> int:
    """Run ONE forecast arm at an unanchored timestep, to mint its evidence.

    The gate and the mint are a chicken and egg -- the config refuses to
    build a forecast at a timestep with no anchor, and an anchor's
    integration half IS two forecasts at that timestep.  This is the only
    path through, it is the same shape ``regional_admission`` uses for its
    contract half, and every part of it is loud: the caller repeats
    :data:`hexcore.dt_admission.CANDIDATE_MINT_AUTHORIZATION` verbatim, the
    schedule receipt must mint clean before a card is touched, the admitted
    row is stamped ``CANDIDATE-UNANCHORED`` and lands stamped that way in the
    run's own receipt, and the admission is withdrawn when the arm exits.

    Two arms of this, byte-identical under masked digests, are the evidence.
    Registering the anchor they earn is a ruling.
    """

    from hexcore import convection_admission, mesh_row_candidate
    from hexcore import pbl_cadence as pbl_cadence_module
    from hexcore.cli import build_parser

    parser = build_parser()
    arguments = parser.parse_args(["forecast", *forecast_argv])
    mesh = getattr(arguments, "mesh", None)
    if not mesh:
        raise MintError(
            "a candidate forecast must name --mesh: the arm's timestep is "
            "declared by a registry row, and without one the door has no "
            "schedule to compute and the anchor has no hardware/mesh pair to "
            "record"
        )

    # An anchor certifies a CONFIGURATION at a timestep, so the candidate
    # admission has to be opened for the configuration the arm will actually
    # run.  Derived from the SAME two inputs the door derives it from -- the
    # row's declared spacing and this run's --convection request -- so the
    # mint and the door cannot open and check different doors.
    import mpas_mesh_binding  # noqa: PLC0415 - tools/ is on sys.path above

    row = mpas_mesh_binding.MESH_BINDINGS.get(mesh)
    if row is None:
        raise MintError(
            f"a candidate forecast must name a REGISTERED mesh; {mesh!r} is "
            f"not one of {sorted(mpas_mesh_binding.MESH_BINDINGS)}"
        )
    decision = convection_admission.convection_decision(
        nominal_dx_m=float(row.nominal_dx_m),
        requested=getattr(arguments, "convection", "auto"),
    )
    cumulus_scheme = decision["constructor_scheme"]

    # The surface/PBL cadence is the other half of the same configuration,
    # and it is derived from the SAME two inputs the door derives it from --
    # the row's declared timestep and this run's --pbl-cadence request -- so
    # the mint and the door cannot open and check different doors.  Without
    # it a held-cadence arm would open a candidate admission for the WELDED
    # configuration and then be refused by the door it just opened.
    pbl_decision = pbl_cadence_module.pbl_cadence_decision(
        dt_seconds=float(dt_seconds),
        requested=getattr(arguments, "pbl_cadence", "auto"),
    )
    surface_pbl_seconds = float(pbl_decision["surface_pbl_seconds"])

    with dt_admission.candidate_mint(
        dt_seconds,
        authorization=authorization,
        card=card,
        cumulus_scheme=cumulus_scheme,
        cumulus_seconds=None if cumulus_scheme is None else float(dt_seconds),
        surface_pbl_seconds=surface_pbl_seconds,
    ) as candidate:
        # The row half of the same chicken and egg: the door takes its
        # timestep from the registry row, so admitting the timestep is not
        # enough on its own.  Opened INSIDE the timestep admission and
        # refused outside it, so a row can never declare a timestep the
        # timestep gate has not been opened for by the same caller.
        with mesh_row_candidate.candidate_mesh_dt(
            mesh,
            dt_seconds,
            authorization=authorization,
            cumulus_scheme=cumulus_scheme,
            surface_pbl_seconds=surface_pbl_seconds,
        ):
            print(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "candidate_mint": candidate.as_dict(),
                        "candidate_mesh_row": {
                            "mesh": mesh,
                            "declared_dt_seconds": float(dt_seconds),
                            "restored_on_exit": True,
                        },
                        "convection": decision,
                        "pbl_cadence": pbl_decision,
                        "warning": (
                            "this arm runs at a timestep holding NO anchor; its "
                            "output is EVIDENCE FOR A RULING, not a forecast "
                            "anybody may quote as anchored"
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return int(arguments.handler(arguments))


def integration_plan(dt_seconds: float, mesh: str, hours: float = 6.0) -> dict[str, Any]:
    """The runnable procedure for the half that needs a card.

    Printed rather than run: the forecast door is the execution path, and a
    second one here would be a second thing to keep true.
    """

    receipt = dt_admission.schedule_receipt(
        dt_seconds, run_steps=max(1, int(round(hours * 3600.0 / float(dt_seconds))))
    )
    steps = int(round(hours * 3600.0 / float(dt_seconds)))
    stride_minutes = receipt["cadences"]["stepra"] * float(dt_seconds) / 60.0
    forecast_arguments = (
        "--mesh {mesh} --grid <grid.nc> --static <static.nc> --init <init.nc> "
        "--init-source '<what produced the init>' --hours {hours:g} "
        "--history-every-minutes {stride:g} --out <arm>/ --scratch <arm>-scratch/"
    ).format(mesh=mesh, hours=hours, stride=max(1.0, stride_minutes))
    command = (
        "python tools/mint_dt_anchor.py --candidate-forecast --dt {dt:g} "
        "--card '<card>' --authorization '{auth}' -- {arguments}"
    ).format(
        dt=float(dt_seconds),
        auth=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        arguments=forecast_arguments,
    )
    return {
        "schema": SCHEMA,
        "dt_seconds": float(dt_seconds),
        "mesh": mesh,
        "hours": hours,
        "steps": steps,
        "arms": 2,
        "procedure": [
            "1. register the row's dt in tools/mpas_mesh_binding.MESH_BINDINGS "
            "(table work; the row must already pass Courant, dual-edge and "
            "radiation-cadence admission)",
            "2. run arm A THROUGH THE CANDIDATE MINT, because the config "
            "refuses an unanchored timestep and that refusal is correct: "
            + command.replace("<arm>", "arm-a"),
            "3. run arm B, same inputs, separate process, same wrapper: "
            + command.replace("<arm>", "arm-b"),
            "4. compare every history file by masked SHA-256 (the netCDF "
            "file_id nonce NUL'd, as the native authority pins do); every "
            "file must match",
            "5. confirm the driver reported finite state at every one of the "
            f"{steps} steps and rc 0 in both arms",
            "6. mint this dt's schedule receipt into the evidence folder: "
            f"python tools/mint_dt_anchor.py --dt {float(dt_seconds):g} "
            "--out evidence/<campaign>/schedule-receipt-dt<dt>.json",
            "7. bring the pair and the receipt to a RULING.  Registering the "
            "anchor moves the frozen lane off its proven timestep and is not "
            "an agent's decision",
        ],
        "what_this_does_not_prove": [
            "byte-identity against native MPAS-A: the only native v8.4.1 "
            "reference this program holds was integrated at "
            f"{dt_admission.PROVEN_DT_SECONDS:g} s, so no other timestep can "
            "have that half without a fresh native run",
            "that the physics is RIGHT at this timestep: Grell-Freitas is "
            "called every step by WRF's own cudt=0 law, so a smaller dt calls "
            "it proportionally more often, and only obs-skill can referee that",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="mint or verify a model-timestep anchor for the frozen lane"
    )
    parser.add_argument("--dt", type=float, help="mint this timestep's schedule receipt")
    parser.add_argument(
        "--radiation-seconds",
        type=float,
        default=dt_admission.RADIATION_CADENCE_SECONDS,
    )
    parser.add_argument("--surface-pbl-seconds", type=float, default=None)
    parser.add_argument("--cumulus-seconds", type=float, default=None)
    parser.add_argument(
        "--cumulus-scheme",
        choices=("gf", "off"),
        default="gf",
        help=(
            "the cumulus selection this receipt is minted for.  'off' is "
            "the 2026-08-26 ruling below 3 km, and it is a DIFFERENT "
            "configuration: it holds its own anchor row and reports no "
            "cumulus cadence at all"
        ),
    )
    parser.add_argument("--run-steps", type=int, default=720)
    parser.add_argument("--out", type=Path, help="write the receipt here")
    parser.add_argument(
        "--verify", action="store_true", help="certify every registered anchor"
    )
    parser.add_argument(
        "--mutation-control",
        action="store_true",
        help="prove the verifier fails on fabricated evidence",
    )
    parser.add_argument(
        "--integration-plan",
        action="store_true",
        help="print the card half's runnable procedure (needs --dt and --mesh)",
    )
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument(
        "--candidate-forecast",
        action="store_true",
        help=(
            "run ONE forecast arm at an unanchored --dt to mint its "
            "integration evidence; needs --card and --authorization, and "
            "every forecast argument after a bare --"
        ),
    )
    parser.add_argument("--card", default=None)
    parser.add_argument("--authorization", default=None)
    parser.add_argument(
        "forecast_argv",
        nargs=argparse.REMAINDER,
        help="forecast arguments, after a bare --",
    )
    arguments = parser.parse_args(argv)

    if arguments.candidate_forecast:
        if arguments.dt is None or not arguments.card or not arguments.authorization:
            parser.error("--candidate-forecast needs --dt, --card and --authorization")
        forward = list(arguments.forecast_argv)
        if forward and forward[0] == "--":
            forward = forward[1:]
        if not forward:
            parser.error(
                "--candidate-forecast needs the forecast arguments after a bare --"
            )
        return run_candidate_forecast(
            arguments.dt,
            card=arguments.card,
            authorization=arguments.authorization,
            forecast_argv=forward,
        )

    if not any(
        (
            arguments.dt is not None,
            arguments.verify,
            arguments.mutation_control,
        )
    ):
        parser.error("choose --dt, --verify or --mutation-control")

    payload: dict[str, Any] = {"schema": SCHEMA}
    status = 0

    if arguments.verify:
        payload["registry"] = verify_registry()
        if not payload["registry"]["certified"]:
            status = 1
    if arguments.mutation_control:
        payload["mutation_control"] = mutation_control()
        if not payload["mutation_control"]["has_teeth_both_directions"]:
            status = 1
    if arguments.dt is not None:
        if arguments.integration_plan:
            if not arguments.mesh:
                parser.error("--integration-plan needs --mesh")
            payload["integration_plan"] = integration_plan(
                arguments.dt, arguments.mesh, arguments.hours
            )
        try:
            scheme = None if arguments.cumulus_scheme == "off" else "gf"
            receipt = mint_schedule_receipt(
                arguments.dt,
                radiation_seconds=arguments.radiation_seconds,
                surface_pbl_seconds=arguments.surface_pbl_seconds,
                cumulus_seconds=arguments.cumulus_seconds,
                cumulus_scheme=scheme,
                run_steps=arguments.run_steps,
            )
        except (DtAdmissionError, MintError) as error:
            payload["schedule_receipt"] = {
                "dt_seconds": arguments.dt,
                "minted": False,
                "refusal": str(error),
            }
            status = 1
        else:
            payload["schedule_receipt"] = receipt
            # Ask about the CONFIGURATION this receipt was minted for, not
            # the timestep alone.  Reading the welded row here would report
            # "anchored" for a receipt minted at a HELD surface/PBL cadence,
            # which is the wrong-row trap the registry key exists to close.
            minted_bldt = float(receipt["cadences"]["surface_pbl_seconds"])
            anchored = (
                dt_admission.admitted_timestep(arguments.dt, scheme, minted_bldt)
                is not None
            )
            payload["anchored_today"] = anchored
            if not anchored:
                payload["registration_note"] = dt_admission.unanchored_refusal(
                    arguments.dt, scheme, minted_bldt
                )
            if arguments.out is not None:
                arguments.out.parent.mkdir(parents=True, exist_ok=True)
                arguments.out.write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                payload["written"] = str(arguments.out)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
