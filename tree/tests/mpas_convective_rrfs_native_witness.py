"""Read-only ptrace witness of the unchanged published native executable.

This traces only its own child. It never writes traced memory/registers, strips
options, modifies the source, or interprets model output. Signal delivery is
preserved. A debugger receipt is failure evidence, not a numeric parity proof.
"""
import argparse, ctypes, hashlib, json, os, resource, signal, sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a=p.parse_args()
binary=a.binary.resolve(strict=True)
libc=ctypes.CDLL(None, use_errno=True)
libc.ptrace.restype=ctypes.c_long
libc.ptrace.argtypes=(ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
def trace(request,pid,addr=0,data=0):
    ctypes.set_errno(0)
    rc=libc.ptrace(request,pid,ctypes.c_void_p(addr),ctypes.c_void_p(data))
    if rc == -1 and ctypes.get_errno(): raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
    return rc
names='r15 r14 r13 r12 rbp rbx r11 r10 r9 r8 rax rcx rdx rsi rdi orig_rax rip cs eflags rsp ss fs_base gs_base ds es fs gs'.split()
class Registers(ctypes.Structure): _fields_=[(n,ctypes.c_ulonglong) for n in names]
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
pid=os.fork()
if pid==0:
    try:
        trace(0,0)
        os.execv(str(binary),[str(binary)])
    except BaseException as error:
        print('trace child failed:',repr(error),file=sys.stderr)
        os._exit(125)
result={'schema':'mpas.rrfs-native-fault-trace/v1','binary':str(binary),
        'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'host':os.uname().nodename,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'rank':os.environ.get('PMI_RANK','singleton'),
        'scope':'read-only debugger on pristine strict binary; borrowed v842 input; no numeric acceptance',
        'memory_or_register_writes':False,'faults':[]}
def save():
    rank=os.environ.get('PMI_RANK','0')
    path=a.out.with_name(a.out.stem+'-rank'+rank+a.out.suffix)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
first_stop=True
try:
    while True:
        _,status=os.waitpid(pid,0)
        if os.WIFEXITED(status): result['exit_status']=os.WEXITSTATUS(status); break
        if os.WIFSIGNALED(status): result['exit_signal']=os.WTERMSIG(status); break
        sig=os.WSTOPSIG(status)
        if sig==signal.SIGSEGV:
            regs=Registers();trace(12,pid,0,ctypes.addressof(regs))
            maps=Path(f'/proc/{pid}/maps').read_text()
            row=next(line for line in maps.splitlines() if str(binary) in line and int(line.split()[2],16)==0)
            base=int(row.split('-')[0],16)
            fault={'registers':{n:hex(getattr(regs,n)) for n in names},
                   'binary_offset':hex(regs.rip-base),'load_base':hex(base),'maps':maps}
            # Stack offsets are decoded only for this authenticated published
            # 51879f7a binary and its observed update+0x3ee0 instruction.
            if result['binary_sha256']=='0b822cc7e2d76455a056f8914609cdc740b10ec5722fe0b7a22a3955f8087739' and regs.rip-base==0x2cc6e0:
                with open(f'/proc/{pid}/mem','rb',buffering=0) as mem:
                    def read(addr,size): mem.seek(addr);return mem.read(size)
                    fault['instruction_hex']=read(regs.rip,4).hex()
                    fault['u10_descriptor_hex']=read(regs.rsp+0x5a0,64).hex()
                    fault['u10_base_address']=hex(int.from_bytes(read(regs.rsp+0x5a0,8),'little'))
                    fault['u10m_mean_descriptor_hex']=read(regs.rsp+0x5e0,64).hex()
                    fault['u10m_mean_base_address']=hex(int.from_bytes(read(regs.rsp+0x5e0,8),'little'))
                    fault['is_needed_u10m_mean']=int.from_bytes(read(base+0xa6d300,4),'little')
                fault['source_inference']='u10m_mean += u10m/nsteps; verify descriptor call and pool field string disassembly separately'
            result['faults'].append(fault)
            # A failed MPI rank may terminate its peers before their exit stop.
            save()
        deliver=0 if first_stop and sig==signal.SIGTRAP else sig
        first_stop=False
        trace(7,pid,0,deliver)
finally:
    # Only metadata and a small debugger register/descriptor witness are saved.
    save()
raise SystemExit(result.get('exit_status',128+result.get('exit_signal',0)))
