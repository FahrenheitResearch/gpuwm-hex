"""Observe the accepted native executable's automatic velCellp, without repair.

The breakpoint is immediately after the conditional 'cell above' assignment,
mpas_pv_diagnostics.F:1007-1012, in the SHA-pinned GNU strict executable. Parent
ptrace restores the original instruction after each observation. Only our child
is traced; the native binary on disk is never changed.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import signal

SHA='a71feef594ae020f86aa9450608e21b8e2074415479de521438a7701e5b1d479'
BREAK_OFFSET=0x238ccb
class Registers(C.Structure):
    _fields_=[(n,C.c_ulonglong) for n in ('r15','r14','r13','r12','rbp','rbx','r11','r10','r9','r8','rax','rcx','rdx','rsi','rdi','orig_rax','rip','cs','eflags','rsp','ss','fs_base','gs_base','ds','es','fs','gs')]
class Floats(C.Structure):
    _fields_=[('control',C.c_ushort*4),('rip',C.c_ulonglong),('rdp',C.c_ulonglong),('mxcsr',C.c_uint),('mask',C.c_uint),('st',C.c_uint*32),('xmm',C.c_uint*64),('padding',C.c_uint*24)]

def trace(native,directory,mode="top"):
    assert hashlib.sha256(native.read_bytes()).hexdigest()==SHA,'native pin changed; disassembly locations require reconciliation'
    libc=C.CDLL(None,use_errno=True);libc.ptrace.restype=C.c_long
    def ptrace(request,pid,address=0,data=0):
        C.set_errno(0)
        result=libc.ptrace(C.c_uint(request),C.c_uint(pid),C.c_void_p(address),C.c_void_p(data))
        if result==-1 and C.get_errno(): raise OSError(C.get_errno(),os.strerror(C.get_errno()))
        return result
    pid=os.fork()
    if pid==0:
        os.chdir(directory)
        fd=os.open('trace-'+os.environ.get('PMI_RANK','0')+'.log',os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o644)
        os.dup2(fd,1);os.dup2(fd,2)
        ptrace(0,0)
        os.execl(str(native),str(native))
    observations=[]
    _,status=os.waitpid(pid,0)
    assert os.WIFSTOPPED(status)
    maps=Path(f'/proc/{pid}/maps').read_text().splitlines()
    base=next(int(l.split('-')[0],16) for l in maps if str(native) in l and l.split()[2]=='00000000')
    bp=base+(BREAK_OFFSET if mode=="top" else 0x237280)
    original=ptrace(1,pid,bp)&((1<<64)-1)
    ptrace(4,pid,bp,(original&~255)|0xcc)
    ptrace(7,pid)
    hits=0
    try:
        while True:
            _,status=os.waitpid(pid,0)
            if os.WIFEXITED(status): return dict(native_sha256=SHA,exit=os.WEXITSTATUS(status),observations=observations)
            if os.WIFSIGNALED(status): raise RuntimeError(f'native child signal {os.WTERMSIG(status)}')
            sig=os.WSTOPSIG(status)
            regs=Registers();ptrace(12,pid,data=C.addressof(regs))
            if sig!=signal.SIGTRAP or regs.rip!=bp+1:
                ptrace(7,pid,data=0 if sig==signal.SIGTRAP else sig);continue
            hits+=1
            fp=Floats();ptrace(14,pid,data=C.addressof(fp))
            if mode=="prefix":
                observations.append(dict(hit=hits,nCells=ptrace(1,pid,regs.rdi)&0xffffffff,
                    nVertLevels=ptrace(1,pid,regs.rsi)&0xffffffff,
                    field0_prefix=ptrace(1,pid,regs.r8-4)&0xffffffff,
                    field1_prefix=ptrace(1,pid,regs.r9-4)&0xffffffff))
            nz=ptrace(1,pid,regs.rsp+0xb8)&0xffffffff
            level=regs.r13&0xffffffff
            if mode=="top" and level>=nz-1:
                p=[ptrace(1,pid,regs.rsp+i)&0xffffffff for i in (0x1a8,0x184,0xbc)]
                cur=[fp.xmm[4*i] for i in (1,2,10)]
                observations.append(dict(hit=hits,level=level,nVertLevels=nz,current_velocity_words=cur,velCellp_words=p))
            ptrace(4,pid,bp,original);regs.rip=bp;ptrace(13,pid,data=C.addressof(regs))
            if len(observations)>=6:
                # Restore/detach, executing the original instruction normally.
                ptrace(17,pid);_,status=os.waitpid(pid,0)
                assert os.WIFEXITED(status)
                return dict(native_sha256=SHA,breakpoint_offset=hex(BREAK_OFFSET if mode=="top" else 0x237280),source='mpas_pv_diagnostics.F:1007-1033' if mode=='top' else 'mpas_pv_diagnostics.F:899,928',exit=os.WEXITSTATUS(status),observations=observations)
            ptrace(9,pid);os.waitpid(pid,0)
            ptrace(4,pid,bp,(original&~255)|0xcc);ptrace(7,pid)
    except BaseException:
        os.kill(pid,signal.SIGKILL);os.waitpid(pid,0);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('native',type=Path);p.add_argument('directory',type=Path);p.add_argument('receipt',type=Path);p.add_argument('--mode',default='top',choices=['top','prefix']);a=p.parse_args()
    r=trace(a.native.resolve(),a.directory.resolve(),a.mode);receipt=a.receipt.with_name(a.receipt.stem+'-rank'+os.environ['PMI_RANK']+a.receipt.suffix) if 'PMI_RANK' in os.environ else a.receipt;receipt.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
