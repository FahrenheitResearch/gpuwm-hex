"""Dimension binding must not change the ABI, arithmetic or tail predicate."""
import pytest
from hexcore.cuda_shape import specialize, resolve

SOURCE = '''extern "C" __global__ void sample(const int nlev, const int ncells,
 const float* x, float* y) {
 int c=blockDim.x*blockIdx.x+threadIdx.x;
 if(c>=ncells) return;
 for(int k=0;k<nlev;++k) y[k*ncells+c]=x[k*ncells+c]*2.0f;
}'''

def test_body_and_abi_preserved():
    code, suffix = specialize(SOURCE, 'sample', (('nlev',55),('ncells',163842)))
    assert 'const int hex_shape_arg_nlev, const int hex_shape_arg_ncells,' in code
    assert 'constexpr int nlev = 55;' in code
    assert code.split('constexpr int ncells = 163842;',1)[1] == SOURCE.split('{',1)[1]
    assert suffix.startswith('.shape_')
    assert specialize(SOURCE, 'sample', ()) == (SOURCE, '')
    assert specialize(SOURCE, 'sample', (('nlev',54),))[1] != suffix

@pytest.mark.parametrize('dims',[(('x',5),),(('nlev',0),),(('nlev',True),),(('nlev',2**31),),(('nlev',2.5),),(('nlev',2),('nlev',3))])
def test_bad_dimension_refused(dims):
    with pytest.raises((ValueError,TypeError)):
        specialize(SOURCE,'sample',dims)

def test_cache_receives_distinct_source_and_module_key():
    class Cache:
        def raw_kernel(self,name,source,*,module_key):return name,source,module_key
    cache=Cache()
    generic=resolve(cache,'sample',SOURCE,'hexcore.sample')
    bound=resolve(cache,'sample',SOURCE,'hexcore.sample',(('nlev',55),))
    assert generic == ('sample',SOURCE,'hexcore.sample')
    assert bound[0] == generic[0] and bound[1:] != generic[1:]

@pytest.mark.parametrize('unit,name,dims',[
 ('cuda_acoustic_v841','acoustic_ru_v841',(('nlev',55),('nedges',491520),('ncells',163842))),
 ('cuda_acoustic_v841','acoustic_rs_ts_chunk_v841',(('nlev',55),('ncells',163842),('nedges',491520),('max_edges',6))),
 ('cuda_dynamics_v841','vector_momentum_hoist_v841_f32',(('nlev',55),('ncells',163842),('nedges',491520),('max_edges2',10))),
 ('cuda_transport_v841','transport_edge_values_hoist_v841',(('ntracers',6),('nlev',55),('ncells',163842),('nedges',491520),('width',10))),
 ('cuda_horizontal_v841','vertex_diagnostics_v841_f32',(('nlev',55),('nedges',491520),('nvertices',327680),('vertex_degree',3))),
 ('partition_overlap_v841','divergence_damping_selected_f32',(('nlev',55),('ncells',163842),('nedges',491520))),
])
def test_real_signatures_body_unchanged(unit,name,dims):
    import importlib,re
    source=importlib.import_module('hexcore.'+unit)._CUDA_SOURCE
    code,_=specialize(source,name,dims)
    # Undo only the explicit signature renames and inserted declarations.
    for dimension,value in dims:
        code=code.replace('hex_shape_arg_'+dimension,dimension)
    declarations='\n'+'\n'.join(f'    constexpr int {d} = {v};' for d,v in dims)
    assert code.replace(declarations,'',1)==source


def test_module_dispatch_binds_shapes_once():
    import importlib
    class Cache:
        def __init__(self):self.calls=[]
        def raw_kernel(self,name,source,*,module_key):
            result=object();self.calls.append((name,source,module_key,result));return result
    for unit,name in [('cuda_acoustic_v841','acoustic_ru_v841'),
                      ('cuda_dynamics_v841','vector_momentum_hoist_v841_f32'),
                      ('cuda_transport_v841','transport_edge_values_hoist_v841')]:
        module=importlib.import_module('hexcore.'+unit);cache=Cache()
        generic=module._kernel(name,cache)
        first=module._kernel(name,cache,(('nlev',55),))
        assert module._kernel(name,cache,(('nlev',55),)) is first
        second=module._kernel(name,cache,(('nlev',54),))
        assert first is not second and first is not generic
        assert len(cache.calls)==3
        assert cache.calls[0][1]==module._CUDA_SOURCE
        assert cache.calls[0][2]=='hexcore.'+unit

def test_boolean_cannot_reuse_a_valid_integer_specialization():
    specialize(SOURCE, 'sample', (('nlev', 1),))
    with pytest.raises(ValueError, match='positive int32'):
        specialize(SOURCE, 'sample', (('nlev', True),))
