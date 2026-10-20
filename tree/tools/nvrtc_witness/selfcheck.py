"""CPU-only real NVRTC witness fixture: parent, spawn, bare -c, fork, failure."""
import multiprocessing
import os
import subprocess
import sys
from cupy_backends.cuda.libs import nvrtc


def compile_one(source='extern "C" __global__ void witness() {}'):
    program = nvrtc.createProgram(source, 'witness.cu', [], [])
    try:
        nvrtc.compileProgram(program, ['--gpu-architecture=sm_120'])
    finally:
        nvrtc.destroyProgram(program)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'nonzero':
        compile_one()
        child = multiprocessing.get_context('spawn').Process(target=compile_one)
        child.start(); child.join()
        assert child.exitcode == 0
        subprocess.run([sys.executable, '-c', 'from cupy_backends.cuda.libs import nvrtc; p=nvrtc.createProgram("", "compile_platform_probe.cu", [], []); nvrtc.compileProgram(p, []); nvrtc.destroyProgram(p)'], check=True)
        child = multiprocessing.get_context('fork').Process(target=compile_one)
        child.start(); child.join()
        assert child.exitcode == 0
        try:
            compile_one('this is deliberately invalid C++')
        except nvrtc.NVRTCError:
            pass
        else:
            raise AssertionError('compiler accepted invalid source')
    elif mode == 'zero':
        subprocess.run([sys.executable, '-c', 'from cupy_backends.cuda.libs import nvrtc; assert nvrtc.getVersion()'], check=True)
    elif mode == 'bypass':
        subprocess.run([sys.executable, '-S', '-c', 'pass'], check=True)
    elif mode == 'replace':
        nvrtc.compileProgram = lambda *args: None
    else:
        raise ValueError(mode)
