"""Compile the production arithmetic helpers without acquiring a GPU.

This checks the defect directly: fast multiply/add must be contractible,
while strict mode retains separate rounded operations and NaN payload tests.
It does not substitute for a column replay or forecast timing.
"""
import re

import pytest

from hexcore import numerics
from hexcore.mpasphys import thompson_v842 as thompson


def _compile_arithmetic(support, mode):
    nvrtc = pytest.importorskip("cupy_backends.cuda.libs.nvrtc")
    # Only the scalar helpers are needed to reproduce this compiler defect.
    support = support[:support.index("// MPAS physics_mmm/mp_radar.F90")]
    aliases = "\n".join(f"#define mpas_{name} {name}" for name in (
        "expf", "exp", "logf", "log", "log10f", "log10", "powf", "pow", "atan", "atanf"))
    source = aliases + "\n" + support + r'''
extern "C" __global__ void arithmetic(float* out, const float* a,
                                    const float* b, const float* c) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    out[i]=tc_add(tc_mul(a[i],b[i]),c[i]);
}
'''
    program = nvrtc.createProgram(source, "thompson_arithmetic.cu", (), ())
    try:
        nvrtc.compileProgram(program, (
            "--gpu-architecture=compute_89",
            *numerics.options(thompson.COMPILE_OPTIONS, numerics=mode)))
        return nvrtc.getPTX(program).decode()
    finally:
        nvrtc.destroyProgram(program)


def test_fast_arithmetic_allows_fma(monkeypatch):
    monkeypatch.setattr(numerics, "mode", lambda: "fast")
    source = thompson.cuda_source(device_math_source="// arithmetic compiler check", max_levels=55)
    ptx = _compile_arithmetic(source, "fast")
    assert re.search(r"\bfma\.rn(?:\.ftz)?\.f32\b", ptx)


def test_strict_arithmetic_keeps_rounding_and_nan_payloads(monkeypatch):
    monkeypatch.setattr(numerics, "mode", lambda: "strict")
    source = thompson.cuda_source(device_math_source="// arithmetic compiler check", max_levels=55)
    ptx = _compile_arithmetic(source, "strict")
    assert "fma.rn" not in ptx
    assert "mul.rn.f32" in ptx
    assert "add.rn.f32" in ptx
    # Classification and payload quieting survive compilation in strict.
    assert "0x7FFFFFFF" in ptx or "2147483647" in ptx
    assert "0x400000" in ptx or "4194304" in ptx


@pytest.mark.parametrize("selected,process", [("strict", "fast"), ("fast", "strict")])
def test_offline_source_mode_overrides_process_mode(monkeypatch, selected, process):
    # A pack writes both modes in one process. Strict rows must not inherit
    # the default process's fast helpers, nor fast rows its strict helpers.
    monkeypatch.setattr(numerics, "mode", lambda: process)
    source = thompson.cuda_source(device_math_source="// offline pack", max_levels=55,
                                  workspace=True, numerics=selected)
    assert ("float r=__fadd_rn(a,b)" in source) == (selected == "strict")
    assert ("__forceinline__ float tc_add" in source) == (selected == "fast")


def test_offline_pack_records_matching_thompson_source_and_options(monkeypatch):
    from hexcore.cuda_backend.aot_build import physics_inventory
    monkeypatch.setattr(numerics, "mode", lambda: "fast")
    for selected in ("strict", "fast"):
        rows = [row for row in physics_inventory(selected)
                if row["physics_family"] == "thompson_workspace"]
        assert len(rows) == 2  # ordinary and WB3 correction variants
        for row in rows:
            assert ("float r=__fadd_rn(a,b)" in row["source"]) == (selected == "strict")
            expected = "--fmad=false" if selected == "strict" else "--fmad=true"
            assert expected in row["options"]
