#!/usr/bin/env python3
"""Count selected native function entries in an owned single-thread MPI rank.

This is execution instrumentation only. Linux ptrace restores each original
instruction before single-stepping it; no source, input or weather array is
changed. Counts are activation witnesses, never forecast parity receipts.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys


class Registers(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
        'r15 r14 r13 r12 rbp rbx r11 r10 r9 r8 rax rcx rdx rsi rdi orig_rax rip cs eflags rsp ss fs_base gs_base ds es fs gs'.split()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--symbol', action='append', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    argv = a.command[1:] if a.command[:1] == ['--'] else a.command
    if platform.machine() != 'x86_64' or not argv:
        raise ValueError('this native activation instrument requires Linux x86_64 and an explicit command')
    executable = Path(argv[0]).resolve(strict=True)
    symbols = {}
    for row in subprocess.check_output(['nm', '-a', str(executable)], text=True).splitlines():
        parts = row.split()
        if len(parts) == 3 and parts[2] in a.symbol:
            symbols[parts[2]] = int(parts[0], 16)
    if set(symbols) != set(a.symbol):
        raise ValueError('requested native function symbol is absent')
    pie = 'DYN (' in subprocess.check_output(['readelf', '-h', str(executable)], text=True)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.ptrace.restype = ctypes.c_long
    libc.ptrace.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    def ptrace(request, pid, address=0, data=0):
        ctypes.set_errno(0)
        value = libc.ptrace(request, pid, address, data)
        error = ctypes.get_errno()
        if value == -1 and error:
            raise OSError(error, os.strerror(error))
        return value
    pid = os.fork()
    if pid == 0:
        ptrace(0, 0)  # TRACEME: only this owned child is observed.
        os.kill(os.getpid(), signal.SIGSTOP)
        os.execv(str(executable), [str(executable), *argv[1:]])
        os._exit(127)
    finished = False
    counts = {name: 0 for name in symbols}
    try:
        _, status = os.waitpid(pid, 0)
        if not os.WIFSTOPPED(status):
            raise RuntimeError('owned native child did not stop before exec')
        ptrace(0x4200, pid, 0, 0x00100000)  # SETOPTIONS / EXITKILL.
        ptrace(7, pid)
        _, status = os.waitpid(pid, 0)
        if not os.WIFSTOPPED(status) or os.WSTOPSIG(status) != signal.SIGTRAP:
            raise RuntimeError('owned native exec did not produce its trace stop')
        base = 0
        if pie:
            mappings = (Path('/proc')/str(pid)/'maps').read_text().splitlines()
            mapping = next(row.split() for row in mappings
                           if row.split()[-1] == str(executable) and row.split()[2] == '00000000')
            base = int(mapping[0].split('-')[0], 16)
        addresses = {base+offset: name for name, offset in symbols.items()}
        originals = {address: ptrace(2, pid, address) & ((1 << 64)-1) for address in addresses}
        for address, word in originals.items():
            ptrace(5, pid, address, (word & ~255) | 0xcc)
        ptrace(7, pid)
        while True:
            _, status = os.waitpid(pid, 0)
            if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                finished = True
                exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else 128+os.WTERMSIG(status)
                break
            if not os.WIFSTOPPED(status):
                raise RuntimeError('unexpected owned native process state')
            stop_signal = os.WSTOPSIG(status)
            if stop_signal == signal.SIGTRAP:
                regs = Registers()
                ptrace(12, pid, 0, ctypes.addressof(regs))
                address = regs.rip-1
                if address not in originals:
                    raise RuntimeError('unexpected native trace trap outside selected function entries')
                counts[addresses[address]] += 1
                ptrace(5, pid, address, originals[address])
                regs.rip = address
                ptrace(13, pid, 0, ctypes.addressof(regs))
                ptrace(9, pid)  # SINGLESTEP executes the original instruction.
                _, stepped = os.waitpid(pid, 0)
                if not os.WIFSTOPPED(stepped) or os.WSTOPSIG(stepped) != signal.SIGTRAP:
                    raise RuntimeError('native entry single-step failed')
                ptrace(5, pid, address, (originals[address] & ~255) | 0xcc)
                ptrace(7, pid)
            else:
                ptrace(7, pid, 0, stop_signal)
        with executable.open('rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        a.receipt.write_text(json.dumps({'schema':'mpas.native-function-entry-count/v1',
            'executable':str(executable),'executable_sha256':digest,'command':argv,
            'ranks':1,'counts':counts,'native_exit':exit_code,
            'method':'owned Linux ptrace child; original entry instructions restored before execution'},indent=2)+'\n')
        return exit_code
    finally:
        if not finished:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
            os.waitpid(pid,0)


if __name__ == '__main__':
    raise SystemExit(main())
