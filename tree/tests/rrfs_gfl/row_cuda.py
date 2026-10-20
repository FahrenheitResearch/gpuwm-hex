"""Full original-row CUDA diagnostic of the unchanged whole GFL transcription.

This preserves source reads, including uninitialized automatic storage. It is
not a production launcher or a workaround for native undefined initial state.
Authority: fork GFL/MPAS/module_cu_gfl_mpas.F90:33-1167 at 1d6e370e.
"""
from functools import lru_cache
import hashlib
import numpy as np
from hexcore.mpasphys.rrfs import cu_gfl_bf6a4b0 as gfl


def observe_initial_state(text, columns=False):
    """Diagnostic substitution of captured native words, never a source default.

    Fork MPAS/module_cu_gfl_mpas.F90:116,233-253,515 omits initialization
    of ierrm. Replaying its passively observed entry words isolates that read.
    All other automatic storage and every scientific expression stay intact.
    """
    changes = (
        ('bool fct1d3_error;', 'bool fct1d3_error; const int* observed_ierrm;', 1),
        ('row_dx_first,initial_error};', 'row_dx_first,initial_error,observed_ierrm' +
            ('+column' if columns else '') + '};', 1),
        ('unsigned int* saved_fct_error){', 'unsigned int* saved_fct_error,const int* observed_ierrm){',
            3 if columns else 2),
        (('vertical,saved_fct_error);' if columns else 'row_dx_first,vertical,saved_fct_error);'),
            ('vertical,saved_fct_error,observed_ierrm);' if columns else
             'row_dx_first,vertical,saved_fct_error,observed_ierrm);'), 2 if columns else 1),
        ('ierrm = (int *) D_7964;', '''ierrm = (int *) D_7964;
      // Diagnostic only: exact captured native initial words, no default.
      for(long long observed_i=0; observed_i<size_576; ++observed_i)
        ierrm[observed_i]=workspace.observed_ierrm[observed_i];''', 1),
    )
    for old, new, count in changes:
        if text.count(old) != count:
            raise ValueError('captured-state diagnostic source seam changed: ' + old)
        text = text.replace(old, new)
    return text


def source(observed=False):
    entry = gfl._ENTRY.split('#ifndef GF_HOST_WITNESS')[0]
    for old,new in (
        ('void gfl_column(int column,int nz,', 'void gfl_row(int ncol,int nz,'),
        (' int ims=1,ime=1,', ' int column=0; int ims=1,ime=ncol,'),
        ('ids=1,ide=2,', 'ids=1,ide=ncol+1,'),
        ('its=1,ite=1,', 'its=1,ite=ncol,'),
    ):
        if entry.count(old)!=1:
            raise ValueError('original GFL entry changed: '+old)
        entry=entry.replace(old,new)
    entry += r'''
extern "C" __global__ void gfl_row_cuda(int ncol,int nz,const unsigned long long* ptrs,
 const unsigned long long* strides,const char* pbl,long long pbl_len,char* scratch,
 unsigned long long capacity,unsigned int* diagnostics,unsigned long long* highwater,
 float row_dx_first,const int* vertical,unsigned int* saved_fct_error){
 gfl_row(ncol,nz,ptrs,strides,pbl,pbl_len,scratch,capacity,diagnostics,highwater,
         row_dx_first,vertical,saved_fct_error);
}
'''
    complete = gfl.cuda_source()
    if complete.count(gfl._ENTRY)!=1:
        raise ValueError('complete production entry identity changed')
    text = complete.replace(gfl._ENTRY, entry)
    return observe_initial_state(text) if observed else text


@lru_cache(maxsize=None)
def module(device,observed=False):
    import cupy as cp
    from cupy_backends.cuda.libs import nvrtc
    with cp.cuda.Device(device):
        arch=cp.cuda.Device(device).compute_capability
        program=nvrtc.createProgram(source(observed),'diagnostic_gfl_row.cu',(),())
        try:
            nvrtc.compileProgram(program,gfl.COMPILER_OPTIONS+(f'--gpu-architecture=sm_{arch}',))
            binary=nvrtc.getCUBIN(program)
        except Exception as error:
            raise RuntimeError(nvrtc.getProgramLog(program)) from error
        finally:
            nvrtc.destroyProgram(program)
        result=cp.cuda.function.Module();result.load(binary)
        return result


def run(records,pbl,ncol,nz,observed_ierrm=None):
    import cupy as cp
    if int(records['num_chem'].reshape(-1)[0])!=0:
        raise ValueError('diagnostic must consume a captured fct1d3 SAVE state for smoke')
    fields={n:None if records[n] is None else cp.array(records[n],order='F',copy=True)
            for n in gfl.FIELDS}
    for alias,target in (('gdc','qc_cu'),('gdc2','qi_cu')):
        if fields[alias] is not None:
            fields[alias]=fields[target]
    pointers=cp.asarray(np.array([fields[n].data.ptr if fields[n] is not None else 0
                                for n in gfl.FIELDS],dtype='u8'))
    strides=cp.zeros(len(gfl.FIELDS),dtype=cp.uint64)
    capacity=ncol*(32768*(nz+1)+131072)
    scratch=cp.empty(capacity,dtype=cp.uint8)
    diagnostics=cp.zeros(1,dtype=cp.uint32); highwater=cp.zeros(1,dtype=cp.uint64)
    # These real calls have no chemistry, so fct1d3 cannot execute. This is the
    # source SAVE default, distinct from the untouched automatic workspace.
    saved_error=cp.zeros(1,dtype=cp.uint32)
    vertical=cp.asarray(np.array([1,nz+1,1,nz+1,1,nz],dtype='i4'))
    text=cp.asarray(np.frombuffer(pbl,dtype='u1'))
    observed = observed_ierrm is not None
    if observed and (observed_ierrm.dtype != np.dtype('<i4') or observed_ierrm.shape != (ncol,)):
        raise ValueError('captured native ierrm must contain exactly the original row words')
    state_words=cp.asarray(observed_ierrm) if observed else None
    kernel=module(cp.cuda.runtime.getDevice(),observed).get_function('gfl_row_cuda')
    arguments=(np.int32(ncol),np.int32(nz),pointers,strides,text,np.int64(len(pbl)),
        scratch,np.uint64(capacity),diagnostics,highwater,np.float32(records['dxcell'][0,0]),
        vertical,saved_error)
    kernel((1,),(1,),arguments + ((state_words,) if observed else ()))
    cp.cuda.get_current_stream().synchronize()
    status=int(diagnostics.get()[0]); peak=int(highwater.get()[0])
    if status:
        raise ValueError('full row CUDA source/scratch diagnostic: '+str(status))
    for name in gfl.OUTPUT_FIELDS:
        if fields[name] is not None:
            records[name][...]=cp.asnumpy(fields[name])
    return dict(source_preserved=not observed,capacity=capacity,scratch_highwater=peak,
                diagnostics=status,automatic_storage_initialized=observed,
                captured_ierrm_substituted=observed, scientific_expressions_preserved=True,
                hidden_state_complete=False,
                emitted_source_sha256=hashlib.sha256(source(observed).encode()).hexdigest())
