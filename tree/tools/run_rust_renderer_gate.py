#!/usr/bin/env python3
"""Materialize and execute the unchanged Rust renderer product path.

The authority run deliberately uses two external executables without modifying
either gpuwm checkout or binary:

* a current tracked-source ``rw_wrfbatch`` build renders five dynamic generic
  forecast fields plus terrain;
* the older installed renderer renders terrain as a backwards-compatible
  named-product control and is explicitly recorded as lacking generic slugs.

Only repository-relative source/artifact paths enter the receipt.  External
binary and basemap paths are intentionally omitted; their hashes/availability
and the observable public wrapper contract are recorded instead.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
import zlib

from mpas_port.rust_renderer import (
    FROZEN_X1_2562_REGRID_WEIGHTS_SHA256,
    INTEGRITY_SCOPE,
    RENDERER_CONTRACT,
    RendererCatalog,
    RendererProbe,
    discover_rust_renderer,
    inspect_renderer_products,
    materialize_gfs_rust_input,
    render_catalogued_products,
    sha256_file,
    validate_rust_wrf2d_netcdf,
)


ROOT = Path(__file__).resolve().parents[1]
GFS_STEM = "GFS-2026-03-26-00.x1.2562.python-port-6h"
DEFAULT_HISTORY = ROOT / "artifacts" / "gfs-forecast" / f"{GFS_STEM}.history.nc"
DEFAULT_LATLON = ROOT / "artifacts" / "gfs-forecast" / f"{GFS_STEM}.latlon.nc"
DEFAULT_GRID = ROOT / "data" / "meshes" / "x1.2562" / "x1.2562.grid.nc"
DEFAULT_STATIC = ROOT / "data" / "meshes" / "x1.2562" / "x1.2562.static.nc"
DEFAULT_REGRID_WEIGHTS = (
    ROOT / "data" / "regrid" / "x1.2562-to-2deg-idw4-renderer-v1.npz"
)
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "rust-renderer"
DEFAULT_ADAPTER = DEFAULT_ARTIFACT_DIR / f"{GFS_STEM}.rw-wrf2d.nc"
DEFAULT_RECEIPT = ROOT / "receipts" / "rust-renderer" / f"{GFS_STEM}.rust-renderer.json"
DEFAULT_CHECKSUMS = ROOT / "receipts" / "rust-renderer" / "SHA256SUMS"
DEFAULT_PRODUCTS = (
    "var:wrf_surface_pressure",
    "var:wrf_temperature_lowest_model_level",
    "var:wrf_u_lowest_model_level",
    "var:wrf_v_lowest_model_level",
    "var:wrf_wind_speed_lowest_model_level",
    "terrain_height",
)
DEFAULT_CURRENT_REVISION = "4152fcb318d7a17ae39967632a788319d64913e3"
DEFAULT_INSTALLED_REVISION = "6a3cbaaf051afd0bd95ea9c1c605fabdbd86dc93"
EXPECTED_CURRENT_RENDERER_SHA256 = (
    "d9e8abeeb622892441f4120fa0b3062ff089ddcaf1f413d9c0817dac61c39983"
)
EXPECTED_CURRENT_RENDERER_BYTES = 9_030_656
EXPECTED_INSTALLED_RENDERER_SHA256 = (
    "c55ec415176cd62c27f63664cf95cf791b91b65648f5a259e9750f4b74d4bae0"
)
EXPECTED_INSTALLED_RENDERER_BYTES = 8_843_264


def repo_relative(path: str | Path) -> str:
    target = Path(path).expanduser().resolve(strict=True)
    try:
        label = target.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            f"receipt path must be inside the repository: {target}"
        ) from error
    if (ROOT / label).resolve(strict=True) != target:
        raise ValueError(f"repository-relative path did not round-trip: {label}")
    return label


def file_record(path: str | Path, *, role: str | None = None) -> dict[str, Any]:
    target = Path(path).expanduser().resolve(strict=True)
    record: dict[str, Any] = {
        "path": repo_relative(target),
        "path_kind": "repo_relative",
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }
    if role is not None:
        record["role"] = role
    return record


def renderer_record(
    probe: RendererProbe,
    *,
    source_revision: str,
    generic_products_supported: bool,
) -> dict[str, Any]:
    return {
        "executable_name": probe.executable.name,
        "executable_path_recorded": False,
        "executable_bytes": probe.executable_bytes,
        "executable_sha256": probe.executable_sha256,
        "source_revision": source_revision,
        "source_revision_verification": (
            "exact revision bytes are embedded in the size/SHA-bound executable"
        ),
        "contract": RENDERER_CONTRACT,
        "probe_evidence": probe.probe_evidence,
        "basemap_available": True,
        "basemap_path_recorded": False,
        "abi_claimed": False,
        "abi_command_invoked": False,
        "abi_note": (
            "rw_wrfbatch exposes no --abi command; gpuwm.rustwx uses --help and "
            "the real-import store catalog, so no ABI version is claimed"
        ),
        "generic_products_supported": generic_products_supported,
    }


def require_frozen_renderer(
    probe: RendererProbe,
    *,
    lane: str,
    expected_sha256: str,
    expected_bytes: int,
    expected_revision: str,
) -> None:
    if (
        probe.executable_sha256 != expected_sha256
        or probe.executable_bytes != expected_bytes
    ):
        raise RuntimeError(
            f"{lane} renderer is not the frozen evidence binary: "
            f"sha256={probe.executable_sha256} bytes={probe.executable_bytes}"
        )
    if expected_revision.encode("ascii") not in probe.executable.read_bytes():
        raise RuntimeError(f"{lane} renderer does not embed the frozen source revision")


def rustwx_record() -> dict[str, Any]:
    module = importlib.import_module("gpuwm.rustwx")
    module_path = Path(module.__file__).resolve(strict=True)
    return {
        "module_name": "gpuwm.rustwx",
        "module_filename": module_path.name,
        "module_path_recorded": False,
        "module_bytes": module_path.stat().st_size,
        "module_sha256": sha256_file(module_path),
    }


def catalog_record(
    catalog: RendererCatalog,
    selected_products: tuple[str, ...],
) -> dict[str, Any]:
    selected_rows = []
    for slug in selected_products:
        row = catalog.row(slug)
        if row is None:
            raise RuntimeError(f"catalog omitted selected product {slug}")
        selected_rows.append(
            {
                "slug": row.slug,
                "kind": row.kind,
                "status": row.status,
                "detail": row.detail,
            }
        )
    return {
        "summary": catalog.summary,
        "rows": len(catalog.rows),
        "status_counts": dict(
            sorted(Counter(row.status for row in catalog.rows).items())
        ),
        "selected_rows": selected_rows,
    }


def _png_metadata(path: Path, width: int, height: int) -> dict[str, Any]:
    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR":
        raise RuntimeError(
            f"renderer output is not a structurally recognizable PNG: {path}"
        )
    actual_width = int.from_bytes(payload[16:20], "big")
    actual_height = int.from_bytes(payload[20:24], "big")
    color_type = payload[25]
    if (actual_width, actual_height) != (width, height):
        raise RuntimeError(
            f"renderer PNG dimensions {(actual_width, actual_height)} != {(width, height)}"
        )
    if color_type != 6:
        raise RuntimeError(f"renderer PNG is not RGBA (PNG color type {color_type})")
    offset = 8
    compressed_parts: list[bytes] = []
    while offset < len(payload):
        length = int.from_bytes(payload[offset : offset + 4], "big")
        chunk_type = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        if chunk_type == b"IDAT":
            compressed_parts.append(data)
        offset += length + 12
        if chunk_type == b"IEND":
            break
    scanlines = zlib.decompress(b"".join(compressed_parts))
    stride = actual_width * 4
    previous = bytearray(stride)
    visible_pixels = 0
    first_visible: bytes | None = None
    varied_visible = False
    position = 0
    for _ in range(actual_height):
        filter_type = scanlines[position]
        position += 1
        row = bytearray(scanlines[position : position + stride])
        position += stride
        for index in range(stride):
            left = row[index - 4] if index >= 4 else 0
            above = previous[index]
            upper_left = previous[index - 4] if index >= 4 else 0
            if filter_type == 1:
                row[index] = (row[index] + left) & 0xFF
            elif filter_type == 2:
                row[index] = (row[index] + above) & 0xFF
            elif filter_type == 3:
                row[index] = (row[index] + ((left + above) // 2)) & 0xFF
            elif filter_type == 4:
                estimate = left + above - upper_left
                pa = abs(estimate - left)
                pb = abs(estimate - above)
                pc = abs(estimate - upper_left)
                predictor = (
                    left if pa <= pb and pa <= pc else above if pb <= pc else upper_left
                )
                row[index] = (row[index] + predictor) & 0xFF
            elif filter_type != 0:
                raise RuntimeError(f"renderer PNG has invalid filter {filter_type}")
        for index in range(0, stride, 4):
            pixel = bytes(row[index : index + 4])
            if pixel[3] > 0:
                visible_pixels += 1
                if first_visible is None:
                    first_visible = pixel
                elif pixel != first_visible:
                    varied_visible = True
        previous = row
    if visible_pixels == 0 or not varied_visible:
        raise RuntimeError(f"renderer PNG is blank, transparent, or constant: {path}")
    return {
        "width": actual_width,
        "height": actual_height,
        "png_color_type": color_type,
        "pixel_format": "RGBA",
        "visible_pixels": visible_pixels,
        "nonconstant_visible_pixels": True,
    }


def _output_product(filename: str, products: tuple[str, ...]) -> str:
    normalized = filename.lower()
    for slug in sorted(products, key=len, reverse=True):
        token = slug.lower().split(":", 1)[-1].replace(".", "_")
        if token in normalized:
            return slug
    raise RuntimeError(
        f"could not bind renderer output filename to a selected product: {filename}"
    )


def publish_pngs(
    outputs: tuple[Path, ...],
    output_sha256: tuple[str, ...],
    *,
    artifact_dir: Path,
    lane: str,
    renderer_sha256: str,
    products: tuple[str, ...],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if len(outputs) != len(output_sha256):
        raise RuntimeError(f"{lane} output path/hash inventory length disagrees")
    for source, expected_source_sha256 in zip(outputs, output_sha256, strict=True):
        if sha256_file(source) != expected_source_sha256:
            raise RuntimeError(f"{lane} RendererRun output hash is stale for {source}")
        product = _output_product(source.name, products)
        product_token = product.split(":", 1)[-1].replace(".", "_")
        destination = artifact_dir / f"{lane}-{renderer_sha256[:8]}-{product_token}.png"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        if sha256_file(destination) != expected_source_sha256:
            raise RuntimeError(f"{lane} published PNG differs from RendererRun output")
        record = file_record(destination, role=f"{lane}_rust_renderer_png")
        record["product"] = product
        record.update(_png_metadata(destination, width, height))
        records.append(record)
    records.sort(key=lambda record: str(record["path"]))
    if len({record["product"] for record in records}) != len(products):
        raise RuntimeError(f"{lane} output set does not bind one PNG to each product")
    return records


def write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="ascii", newline="\n")
    temporary.replace(path)


MappingLike = dict[str, Any]


def write_checksums(path: Path, targets: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{sha256_file(target)}  {repo_relative(target)}" for target in targets]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--installed-control-renderer", type=Path)
    parser.add_argument(
        "--generated-utc",
        required=True,
        help="frozen evidence timestamp in YYYY-MM-DDTHH:MM:SSZ form",
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--latlon", type=Path, default=DEFAULT_LATLON)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--regrid-weights", type=Path, default=DEFAULT_REGRID_WEIGHTS)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUMS)
    parser.add_argument("--products", default=",".join(DEFAULT_PRODUCTS))
    parser.add_argument("--frame-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--ztop", type=float, default=30_000.0)
    return parser


def require_frozen_repository_paths(args: argparse.Namespace) -> None:
    """Refuse path substitution before materialization can write anything."""

    frozen = {
        "history": DEFAULT_HISTORY,
        "latlon": DEFAULT_LATLON,
        "grid": DEFAULT_GRID,
        "static": DEFAULT_STATIC,
        "regrid_weights": DEFAULT_REGRID_WEIGHTS,
        "adapter": DEFAULT_ADAPTER,
        "artifact_dir": DEFAULT_ARTIFACT_DIR,
        "receipt": DEFAULT_RECEIPT,
        "checksums": DEFAULT_CHECKSUMS,
    }
    for argument, expected in frozen.items():
        actual = Path(getattr(args, argument)).expanduser().resolve()
        if actual != expected.resolve():
            raise ValueError(
                f"the frozen M5 gate requires --{argument.replace('_', '-')} "
                f"to remain {expected.relative_to(ROOT).as_posix()}"
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_frozen_repository_paths(args)
    started = time.perf_counter()
    products = tuple(part.strip() for part in args.products.split(",") if part.strip())
    if products != DEFAULT_PRODUCTS:
        raise ValueError(
            "the frozen M5 gate requires all five dynamic generic fields plus terrain"
        )
    if args.frame_index != 1:
        raise ValueError("the frozen M5 gate renders the final f006 frame index 1")
    if args.width != 1400 or args.height != 900:
        raise ValueError("the frozen M5 gate requires 1400x900 output")
    if args.ztop != 30_000.0:
        raise ValueError("the frozen M5 gate requires the forecast's 30000 m model top")
    try:
        generated_utc = time.strptime(args.generated_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("--generated-utc must use YYYY-MM-DDTHH:MM:SSZ") from error
    del generated_utc
    history = args.history.expanduser().resolve(strict=True)
    latlon = args.latlon.expanduser().resolve(strict=True)
    grid = args.grid.expanduser().resolve(strict=True)
    static = args.static.expanduser().resolve(strict=True)
    regrid_weights = args.regrid_weights.expanduser().resolve(strict=True)
    adapter = args.adapter.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    checksums_path = args.checksums.expanduser().resolve()
    for path in (history, latlon, grid, static, regrid_weights):
        repo_relative(path)
    for path in (adapter, artifact_dir, receipt_path, checksums_path):
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as error:
            raise ValueError(
                f"output must remain inside the repository: {path}"
            ) from error
    if adapter.parent != artifact_dir:
        raise ValueError("the frozen adapter must be a direct child of artifact-dir")
    for metadata_path in (receipt_path, checksums_path):
        try:
            metadata_path.relative_to(artifact_dir)
        except ValueError:
            pass
        else:
            raise ValueError(
                "receipt and checksum inventory must remain outside artifact-dir"
            )
    source_paths = {history, latlon, grid, static, regrid_weights}
    output_paths = {adapter, receipt_path, checksums_path}
    if len(output_paths) != 3 or source_paths.intersection(output_paths):
        raise ValueError("source and output file paths must be distinct")

    checkpoint = time.perf_counter()
    materialize_gfs_rust_input(
        adapter,
        history_path=history,
        latlon_path=latlon,
        grid_path=grid,
        static_path=static,
        regrid_weights_path=regrid_weights,
        expected_regrid_weights_sha256=FROZEN_X1_2562_REGRID_WEIGHTS_SHA256,
        ztop_m=args.ztop,
        clobber=True,
    )
    adapter_validation = validate_rust_wrf2d_netcdf(
        adapter,
        materialized_sources={
            "history": history,
            "latlon": latlon,
            "grid": grid,
            "static": static,
            "regrid_weights": regrid_weights,
        },
        require_materialized_provenance=True,
    )
    materialize_seconds = time.perf_counter() - checkpoint

    current_probe = discover_rust_renderer(args.renderer)
    control_probe = discover_rust_renderer(args.installed_control_renderer)
    require_frozen_renderer(
        current_probe,
        lane="current tracked-source",
        expected_sha256=EXPECTED_CURRENT_RENDERER_SHA256,
        expected_bytes=EXPECTED_CURRENT_RENDERER_BYTES,
        expected_revision=DEFAULT_CURRENT_REVISION,
    )
    require_frozen_renderer(
        control_probe,
        lane="installed control",
        expected_sha256=EXPECTED_INSTALLED_RENDERER_SHA256,
        expected_bytes=EXPECTED_INSTALLED_RENDERER_BYTES,
        expected_revision=DEFAULT_INSTALLED_REVISION,
    )
    if current_probe.executable_sha256 == control_probe.executable_sha256:
        raise RuntimeError(
            "current generic renderer and installed compatibility control must be distinct binaries"
        )

    with tempfile.TemporaryDirectory(prefix="mpas-rust-renderer-gate-") as temporary:
        work = Path(temporary)
        current_catalog = inspect_renderer_products(
            adapter,
            store_root=work / "current-store",
            probe=current_probe,
        )
        current_run = render_catalogued_products(
            adapter,
            store_root=work / "current-store",
            out_dir=work / "current-out",
            products=products,
            probe=current_probe,
            catalog=current_catalog,
            frames=str(args.frame_index),
            width=args.width,
            height=args.height,
            source_label="MPAS-Atmosphere Python CPU port",
        )
        current_artifacts = publish_pngs(
            current_run.outputs,
            current_run.output_sha256,
            artifact_dir=artifact_dir,
            lane="current",
            renderer_sha256=current_probe.executable_sha256,
            products=products,
            width=args.width,
            height=args.height,
        )

        control_products = ("terrain_height",)
        control_catalog = inspect_renderer_products(
            adapter,
            store_root=work / "installed-store",
            probe=control_probe,
        )
        control_run = render_catalogued_products(
            adapter,
            store_root=work / "installed-store",
            out_dir=work / "installed-out",
            products=control_products,
            probe=control_probe,
            catalog=control_catalog,
            frames=str(args.frame_index),
            width=args.width,
            height=args.height,
            source_label="MPAS Python port - installed renderer terrain control",
        )
        control_artifacts = publish_pngs(
            control_run.outputs,
            control_run.output_sha256,
            artifact_dir=artifact_dir,
            lane="installed-control",
            renderer_sha256=control_probe.executable_sha256,
            products=control_products,
            width=args.width,
            height=args.height,
        )

    adapter_record = file_record(adapter, role="rw_wrfbatch_postprocessed_2d_input")
    artifact_records = [adapter_record, *current_artifacts, *control_artifacts]
    expected_artifact_files = {
        (ROOT / str(record["path"])).resolve() for record in artifact_records
    }
    actual_artifact_files = {
        path.resolve(strict=True) for path in artifact_dir.iterdir() if path.is_file()
    }
    if actual_artifact_files != expected_artifact_files:
        raise RuntimeError(
            "artifact directory has stale or missing files outside the exact evidence inventory"
        )
    current_generic_rows = [
        current_catalog.row(slug) for slug in products if slug.startswith("var:")
    ]
    if any(row is None or row.status != "renderable" for row in current_generic_rows):
        raise RuntimeError(
            "current renderer did not expose every generic forecast product"
        )
    control_generic_rows = [
        row for row in control_catalog.rows if row.slug.startswith("var:")
    ]
    if control_generic_rows:
        raise RuntimeError("installed control unexpectedly exposes generic products")

    receipt: dict[str, Any] = {
        "schema": "mpas-port.rust-renderer-gate.v1",
        "receipt_path": repo_relative(receipt_path.parent / receipt_path.name)
        if receipt_path.exists()
        else receipt_path.relative_to(ROOT).as_posix(),
        "generated_utc": args.generated_utc,
        "evidence": {
            "status": "passed",
            "classification": (
                "real committed GFS-initialized MPAS Python forecast rendered by unchanged "
                "rw_wrfbatch through the mesh-to-latlon postprocessed-2D adapter"
            ),
            "dynamic_forecast_product_rendered": True,
            "terrain_product_rendered": True,
            "non_claim": (
                "the source forecast uses physics_suite=none; visualization regridding is "
                "implemented-unverified and non-conservative; this is execution/product-path "
                "evidence, not forecast-skill or Fortran-equivalence evidence; generic fields "
                "use an automatic full-finite-range style, not an operational meteorological "
                "colortable"
            ),
            "integrity_scope": INTEGRITY_SCOPE,
        },
        "sources": {
            "history": file_record(history),
            "latlon": file_record(latlon),
            "grid": file_record(grid),
            "static": file_record(static),
            "regrid_weights": {
                **file_record(regrid_weights),
                "schema": "mpas-port.saved-regrid-weights.v2",
                "renderer_materialization_schema": (
                    "mpas-port.renderer-materialization-authority.v1"
                ),
                "method": "inverse_distance",
                "neighbors": 4,
                "power": 2.0,
                "selection": (
                    "frozen exact x1.2562-to-2-degree weights plus mesh-derived "
                    "reconstruction geometry; no runtime cKDTree or platform-libm "
                    "geometry recomputation"
                ),
            },
        },
        "adapter": {
            **adapter_record,
            "validation": adapter_validation,
            "truthful_gate_fields": {
                "TK": "temperature at lowest model level",
                "P": "pressure at lowest model level",
                "Z": (
                    "geometric height of the lowest mass level over the default one-pass-"
                    "smoothed terrain; additional vertical-surface smoothing is disabled"
                ),
                "PSFC": "surface pressure",
                "HGT": "static-file terrain height",
            },
            "forbidden_aliases_absent": [
                "PB",
                "T2",
                "U10",
                "V10",
                "SLP",
                "Times",
                "XTIME",
                "xtime",
            ],
        },
        "renderer": renderer_record(
            current_probe,
            source_revision=DEFAULT_CURRENT_REVISION,
            generic_products_supported=True,
        ),
        "renderer_build": {
            "source_revision": DEFAULT_CURRENT_REVISION,
            "declared_authority_build_command": (
                "cargo build --release --locked --offline --package rw-wrfbatch "
                "--bin rw_wrfbatch"
            ),
            "build_command_verified_by_binary_sha": False,
            "external_cargo_target_dir": True,
            "target_path_recorded": False,
            "not_modified_by_this_gate": True,
        },
        "catalog": catalog_record(current_catalog, products),
        "render": {
            "selected_products": list(products),
            "frame_index": args.frame_index,
            "valid_time": "2026-03-26T06:00:00Z",
            "forecast_lead_seconds": 21_600,
            "width": args.width,
            "height": args.height,
            "source_label": "MPAS-Atmosphere Python CPU port",
            "skipped": [],
            "failures": [],
            "png_count": len(current_artifacts),
        },
        "installed_control": {
            "renderer": renderer_record(
                control_probe,
                source_revision=DEFAULT_INSTALLED_REVISION,
                generic_products_supported=False,
            ),
            "catalog": catalog_record(control_catalog, control_products),
            "selected_products": list(control_products),
            "purpose": (
                "the older installed binary predates generic var products but renders true "
                "MPAS terrain through the same adapter"
            ),
            "skipped": [],
            "failures": [],
            "png_count": len(control_artifacts),
        },
        "artifacts": artifact_records,
        "timing_seconds": {
            "scope": (
                "observed wall timings for this frozen run; intentionally nondeterministic "
                "and not a reproducibility claim"
            ),
            "materialize_and_validate": materialize_seconds,
            "current_real_import_catalog": current_catalog.elapsed_seconds,
            "current_render": current_run.elapsed_seconds,
            "installed_real_import_catalog": control_catalog.elapsed_seconds,
            "installed_render": control_run.elapsed_seconds,
            "total_before_receipt": time.perf_counter() - started,
        },
        "tool": {
            "runner_path": repo_relative(Path(__file__)),
            "runner_sha256": sha256_file(Path(__file__)),
            "module_path": "src/mpas_port/rust_renderer.py",
            "module_sha256": sha256_file(
                ROOT / "src" / "mpas_port" / "rust_renderer.py"
            ),
            "gpuwm_not_modified_by_this_gate": True,
            "rustwx_wrapper": rustwx_record(),
        },
    }
    write_json(receipt_path, receipt)
    checksum_targets = [
        receipt_path,
        *[ROOT / record["path"] for record in artifact_records],
    ]
    write_checksums(checksums_path, checksum_targets)
    print(
        json.dumps(
            {
                "status": "passed",
                "receipt": repo_relative(receipt_path),
                "adapter": repo_relative(adapter),
                "current_renderer_sha256": current_probe.executable_sha256,
                "installed_renderer_sha256": control_probe.executable_sha256,
                "dynamic_products": list(products),
                "pngs": len(current_artifacts) + len(control_artifacts),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
