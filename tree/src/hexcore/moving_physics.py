"""Carry resident column physics between meshes and recomputed LTS classes.

Owner columns alone form the source. Halo and scratch columns are rebuilt
from those owners. Integer categories use the largest-overlap donor. Floating
SAVE fields use the same positive overlap operator as the atmosphere.
"""
from copy import deepcopy
import numpy as np

CATEGORIES = {'fields/xland', 'fields/xice', 'fields/landmask', 'fields/lakemask'}
STATIC = {'fields/xland', 'fields/xice', 'fields/landmask', 'fields/lakemask',
          'fields/isltyp', 'fields/ivgtyp', 'fields/shdmax', 'fields/shdmin',
          'fields/tmn', 'fields/vegfra', 'fields/albbck', 'fields/embck'}
# Noah-MP's stores and layer geometry form one column state. Independently
# averaging them while selecting integer snow layers from a donor breaks
# the solver's water accounting. Carry this group from one compatible donor.
LAND_STATE = {'fields/'+name for name in (
    'snow','snowh','canwat','tslb','smois','sh2o','lai',
    'tvxy','tgxy','canicexy','canliqxy','eahxy','tahxy','cmxy','chxy','fwetxy',
    'sneqvoxy','alboldxy','qsnowxy','qrainxy','wslakexy','waxy','xsaixy','taussxy',
    'isnowxy','pgsxy','tsnoxy','snicexy','snliqxy','zsnsoxy')}
LAND_CATEGORIES = ('fields/xland','fields/xice','fields/isltyp','fields/ivgtyp')
VECTOR_PAIRS=(('pbl/du','pbl/dv'),('seam/du','seam/dv'),('fields/u10','fields/v10'))+tuple(
    (f'driver/{scheme}_tendencies/ru',f'driver/{scheme}_tendencies/rv')
    for scheme in ('pbl','radiation','cumulus'))


# Classify by the sealed field contract, never by a coincidental soil/level
# dimension match (reduced tests can have four atmospheric and soil levels).
ATMOSPHERIC_FIELDS = {'diag/cldfra', 'fields/exch_h', 'fields/exch_m',
    'radiation/o33d_grid', 'cumulus/w0avg', 'driver/rthratenlw', 'driver/rthratensw'}
ATMOSPHERIC_PREFIXES = ('pbl/', 'seam/', 'state/', 'driver/pbl_tendencies/',
    'driver/radiation_tendencies/', 'driver/cumulus_tendencies/')


def atmospheric_intensive(values, weights):
    mapper = getattr(weights, 'layer_intensive', weights.intensive)
    return mapper(values)


def field_intensive(values, weights, key):
    if key in ATMOSPHERIC_FIELDS or key.startswith(ATMOSPHERIC_PREFIXES):
        # Known column rates use the mass grid. Fail on an unknown vertical
        # layout rather than silently treating it as computational levels.
        if hasattr(weights, 'source_dzw') and values.shape[0] == len(weights.source_dzw) + 1:
            shape = values.shape
            flat = values.reshape(shape[0], -1, shape[-1])
            mapped = np.stack([weights.interface_intensive(flat[:, j])
                               for j in range(flat.shape[1])], axis=1)
            return mapped.reshape((mapped.shape[0], *shape[1:-1], mapped.shape[-1]))
        return atmospheric_intensive(values, weights)
    return weights.intensive(values)


def local_vector_bases(mesh):
    lat=np.asarray(mesh.arrays['latCell'],dtype=float)
    lon=np.asarray(mesh.arrays['lonCell'],dtype=float)
    east=np.stack((-np.sin(lon),np.cos(lon),np.zeros_like(lon)),axis=-1)
    north=np.stack((-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)),axis=-1)
    return east,north


def carry_horizontal_vector(u,v,weights,source_mesh,target_mesh, *, atmospheric=False):
    if u.dtype!=v.dtype:
        raise ValueError('physics vector components have incompatible dtypes')
    y_stagger=(u.ndim==3 and v.ndim==3 and u.shape[1]==1 and v.shape[1]==2 and
               u.shape[0]==v.shape[0] and u.shape[2]==v.shape[2])
    if y_stagger:
        # The column seam has one physical y column and two duplicated y
        # faces. Verify that contract before collapsing and rebuilding it.
        if not np.array_equal(v[:,0],v[:,1]):
            raise ValueError('physics pseudo y-edge copies differ')
        mapped_u,mapped_v=carry_horizontal_vector(u[:,0],v[:,0],weights,source_mesh,target_mesh,atmospheric=atmospheric)
        return mapped_u[:,None,:],np.repeat(mapped_v[:,None,:],2,axis=1)
    if u.shape!=v.shape:
        raise ValueError(f'physics vector components have incompatible layouts: {u.shape}, {v.shape}')
    east,north=local_vector_bases(source_mesh)
    target_east,target_north=local_vector_bases(target_mesh)
    mapper = (lambda x: atmospheric_intensive(x, weights)) if atmospheric else weights.intensive
    components=[mapper(u*east[:,axis]+v*north[:,axis]) for axis in range(3)]
    mapped_u=sum(value*target_east[:,axis] for axis,value in enumerate(components))
    mapped_v=sum(value*target_north[:,axis] for axis,value in enumerate(components))
    return mapped_u.astype(u.dtype),mapped_v.astype(v.dtype)


def coupled_column_faces(du,dv):
    """Native unit-mass column seam's periodic C-grid interpolation."""
    if du.shape!=dv.shape or du.ndim!=3 or du.shape[1]!=1 or du.dtype!=dv.dtype:
        raise ValueError('raw column PBL rates have incompatible layouts')
    ru=np.float32(.5)*(du+np.roll(du,1,axis=2))
    return np.concatenate((ru,ru[:,:,:1]),axis=2),np.concatenate((dv,dv[:,:1,:]),axis=1)


def gather_owner_columns(arrays, layouts, n_cells):
    shape = arrays[0].shape[:-1]
    result = np.empty((*shape, n_cells), dtype=arrays[0].dtype)
    seen = np.zeros(n_cells, dtype=bool)
    padding = []
    for array, layout in zip(arrays, layouts):
        local = len(layout.cell_l2g)
        if array.shape[:-1] != shape or array.shape[-1] not in (local, local+1):
            raise ValueError('physics field has an unknown column layout')
        count = layout.n_owned_cells
        ids = np.asarray(layout.cell_l2g[:count], dtype=np.int64)
        if np.any(seen[ids]):
            raise ValueError('physics owner columns overlap')
        seen[ids] = True
        result[..., ids] = array[..., :count]
        if array.shape[-1] == local+1:
            padding.append(float(np.max(abs(array[..., -1]))))
    if not np.all(seen):
        raise ValueError('physics owner columns do not cover the globe')
    return result, padding


def largest_donors(weights):
    matrix=weights.overlap
    return np.array([matrix.indices[matrix.indptr[i]+np.argmax(
        matrix.data[matrix.indptr[i]:matrix.indptr[i+1]])]
        for i in range(matrix.shape[0])], dtype=np.int64)


def carry_field(values, weights, *, categorical=False, donors=None):
    if categorical or np.issubdtype(values.dtype, np.integer):
        return values[..., largest_donors(weights) if donors is None else donors]
    return weights.intensive(values).astype(values.dtype)


def mapped_owner_field(payload, weights, key, cache=None, *, target_mesh=None):
    """Map a sealed owner field once, then slice it into every new class."""
    if cache is None:
        cache={}
    identity=(id(payload),id(weights),id(target_mesh))
    if cache.setdefault('identity',identity)!=identity:
        raise ValueError('physics field cache belongs to another remap')
    fields=cache.setdefault('fields',{})
    if key not in fields:
        pair=next((pair for pair in VECTOR_PAIRS if key in pair),None)
        if pair is not None:
            if target_mesh is None:
                raise ValueError('physics vector remap requires target geography')
            old=[gather_owner_columns([source['seam']['arrays'][component] for source in payload['backends']],
                 payload['layouts'],len(weights.source_area)) for component in pair]
            mapped=carry_horizontal_vector(old[0][0],old[1][0],weights,payload['mesh'],target_mesh,
                atmospheric=pair[0] != 'fields/u10')
            for index,component in enumerate(pair):fields[component]=(mapped[index],old[index][1],False)
            return fields[key]
        values,padding=gather_owner_columns(
            [source['seam']['arrays'][key] for source in payload['backends']],
            payload['layouts'],len(weights.source_area))
        categorical=key in CATEGORIES or np.issubdtype(values.dtype,np.integer)
        if categorical and 'donors' not in cache:
            cache['donors']=largest_donors(weights)
        mapped = (carry_field(values,weights,categorical=True,donors=cache.get('donors'))
                  if categorical else field_intensive(values,weights,key).astype(values.dtype))
        fields[key]=(mapped,padding,categorical)
    return fields[key]


def land_state_donors(payload,weights,layout,fresh,target_mesh,cache):
    """One local coherent donor, preserving land, sea-ice and glacier roles."""
    from scipy.spatial import cKDTree
    from .cuda_arwen_physics_v841 import _ISICE_TABLE
    sources=cache.setdefault('land_sources',{})
    for key in LAND_CATEGORIES:
        if key not in sources:
            sources[key]=gather_owner_columns([b['seam']['arrays'][key] for b in payload['backends']],
                                              payload['layouts'],len(weights.source_area))[0]
    categories=np.stack([sources[key].reshape(-1) for key in LAND_CATEGORIES],axis=-1)
    count=len(layout.cell_l2g)
    targets=np.stack([fresh[key].reshape(-1)[:count] for key in LAND_CATEGORIES],axis=-1)
    threshold=float(payload['backends'][0]['seam']['identity']['xice_threshold'])
    categories[:,0]=categories[:,0]>=1.5;targets[:,0]=targets[:,0]>=1.5
    categories[:,1]=categories[:,1]>=threshold;targets[:,1]=targets[:,1]>=threshold
    categories=np.column_stack((categories,categories[:,3]==_ISICE_TABLE))
    targets=np.column_stack((targets,targets[:,3]==_ISICE_TABLE))
    if 'donors' not in cache:cache['donors']=largest_donors(weights)
    defaults=cache['donors']
    ids=defaults[layout.cell_l2g].copy()
    mismatch=np.any(categories[ids]!=targets,axis=1)
    fallback=0;distance_max=0.;relaxed={}
    trees=cache.setdefault('land_category_trees',{})
    for column in np.flatnonzero(mismatch):
        row=int(layout.cell_l2g[column]);start,end=weights.overlap.indptr[row:row+2]
        candidates=weights.overlap.indices[start:end]
        found=False;tier=None
        for fields in ((0,1,2,3,4),(0,1,2,4),(0,1,3,4),(0,1,4)):
            compatible=np.all(categories[candidates][:,fields]==targets[column,list(fields)],axis=1)
            if not np.any(compatible):continue
            positions=np.flatnonzero(compatible)
            best=positions[np.argmax(weights.overlap.data[start:end][positions])]
            ids[column]=candidates[best];tier=fields;found=True;break
        if found:
            if len(tier)<5:
                name=','.join(LAND_CATEGORIES[i] for i in tier if i<4)
                relaxed[name]=relaxed.get(name,0)+1
            continue
        fields=(0,1,4)
        category=tuple(targets[column,list(fields)])
        if category not in trees:
            available=np.flatnonzero(np.all(categories[:,fields]==targets[column,list(fields)],axis=1))
            if not len(available):
                raise ValueError(f'no carried land column matches new land/ice/glacier role {category}')
            coordinates=np.stack([payload['mesh'].arrays[k][available] for k in ('xCell','yCell','zCell')],axis=-1)
            coordinates/=np.linalg.norm(coordinates,axis=1,keepdims=True)
            trees[category]=(available,cKDTree(coordinates))
        available,tree=trees[category]
        point=np.array([target_mesh.arrays[k][row] for k in ('xCell','yCell','zCell')],dtype=float)
        point/=np.linalg.norm(point)
        chord,index=tree.query(point)
        ids[column]=available[index];fallback+=1
        name=','.join(LAND_CATEGORIES[i] for i in fields if i<4)
        relaxed[name]=relaxed.get(name,0)+1
        distance_max=max(distance_max,float(2*np.arcsin(min(chord/2,1))*
                                           payload['mesh'].attrs['sphere_radius']/1000))
    return ids,dict(columns=count,dominant_category_mismatch_columns=int(mismatch.sum()),
                   changed_dominant_donors=int(np.count_nonzero(ids!=defaults[layout.cell_l2g])),
                   nearest_compatible_fallback_columns=fallback,
                   nearest_compatible_fallback_distance_max_km=distance_max,
                   relaxed_category_donor_columns=relaxed,xice_threshold=threshold,
                   categories=list(LAND_CATEGORIES))


def restore_remapped_backend(backend, payload, weights, layout, *, base_dt, rate, mapped_cache=None,target_mesh=None):
    fresh=backend.restart_state()
    sources=payload['backends']
    # Only the sealed column census changes. Never bypass an engine/physics
    # contract mismatch by manufacturing a fresh identity around old data.
    for source in sources:
        if {k:v for k,v in source['identity'].items() if k!='constructor_identity_sha256'} != {
                k:v for k,v in fresh['identity'].items() if k!='constructor_identity_sha256'}:
            raise ValueError('remapped physics engine or contract changed')
        changed={'n_columns','dt','stepra','stepbl','stepcu'}
        old_selection={k:v for k,v in source['seam']['identity'].items() if k not in changed}
        new_selection={k:v for k,v in fresh['seam']['identity'].items() if k not in changed}
        if old_selection != new_selection:
            difference={key:{'source':old_selection.get(key),'target':new_selection.get(key)}
                        for key in old_selection.keys()|new_selection.keys()
                        if old_selection.get(key)!=new_selection.get(key)}
            raise ValueError(f'remapped physics selection changed: {difference}')
    keys=set(sources[0]['seam']['arrays'])
    if any(set(source['seam']['arrays'])!=keys for source in sources):
        raise ValueError('source physics classes have different persistent fields')
    if set(fresh['seam']['arrays'])-keys:
        raise ValueError('target physics needs fields absent from the boundary')
    receipt=[]
    result=dict(fresh['seam']['arrays'])
    local=len(layout.cell_l2g)
    if mapped_cache is None:mapped_cache={}
    land_donors,land_receipt=land_state_donors(payload,weights,layout,fresh['seam']['arrays'],target_mesh,mapped_cache)
    for key in sorted(keys):
        if key in LAND_STATE:
            land_fields=mapped_cache['land_sources']
            if key not in land_fields:
                land_fields[key],padding=gather_owner_columns([b['seam']['arrays'][key] for b in payload['backends']],
                                                          payload['layouts'],len(weights.source_area))
            else:padding=[]
            selected=np.ascontiguousarray(land_fields[key][...,land_donors]);categorical=True
        else:
            mapped,padding,categorical=mapped_owner_field(payload,weights,key,mapped_cache,target_mesh=target_mesh)
            selected=mapped[..., np.asarray(layout.cell_l2g,dtype=np.int64)]
        if key in fresh['seam']['arrays']:
            target=fresh['seam']['arrays'][key]
            if target.shape[:-1]!=selected.shape[:-1] or target.shape[-1] not in (local,local+1):
                raise ValueError(f'unsupported target physics layout: {key}')
            if target.shape[-1]==local+1:
                # Coupled x-momentum has a periodic alias of its first face.
                # Other pseudo columns remain non-owning duplicate padding.
                alias=selected[...,:1] if key.endswith('_tendencies/ru') else selected[...,-1:]
                selected=np.concatenate((selected,alias),axis=-1)
        else:
            # Existing lazily born ozone/cumulus caches are restart-authorized.
            if key not in ('radiation/o33d_grid','cumulus/w0avg'):
                raise ValueError(f'unknown lazy physics carrier: {key}')
        method='largest-overlap donor' if categorical else 'conservative area overlap'
        if key in LAND_STATE:method='local coherent Noah-MP column donor preserving land/ice/glacier role; prefer overlapping soil and vegetation matches'
        if any(key in pair for pair in VECTOR_PAIRS):method='earth Cartesian overlap, target east/north projection'
        if hasattr(weights, 'layer_intensive') and (key in ATMOSPHERIC_FIELDS or key.startswith(ATMOSPHERIC_PREFIXES)):
            method += '; physical-height atmospheric interpolation'
        if key in STATIC:
            selected=fresh['seam']['arrays'][key]
            method='new geography constructor, persistent dynamical fields carried separately'
        result[key]=selected
        receipt.append(dict(field=key,method=method,source_padding_abs_max=padding,
                            shape=list(selected.shape)))
    # The driver's held x-face rates depend on local column order. Rebuild
    # them from the carried physical raw PBL rates, using native interpolation.
    if 'pbl/du' in result and 'pbl/dv' in result:
        ru,rv=coupled_column_faces(result['pbl/du'],result['pbl/dv'])
        for key,value in (('driver/pbl_tendencies/ru',ru),('driver/pbl_tendencies/rv',rv)):
            if key in result:
                if result[key].shape!=value.shape:raise ValueError('rebuilt PBL face cache extent differs')
                result[key]=value
                next(item for item in receipt if item['field']==key)['method']='native face interpolation of remapped raw PBL rates'
    scalars=deepcopy(sources[0]['seam']['scalars'])
    for source in sources:
        for key in ('elapsed_seconds','carriers','kf_history_time'):
            if source['seam']['scalars'][key]!=scalars[key]:
                raise ValueError(f'physics caches have incompatible boundary clocks: {key}')
    step=float(payload['time_seconds'])/(base_dt*rate)
    if step!=int(step):
        raise ValueError('physics boundary does not lie on the new class clock')
    scalars['step_index']=int(step)
    fresh['seam']=dict(identity=fresh['seam']['identity'],arrays=result,scalars=scalars)
    fresh['adapter']=deepcopy(sources[0]['adapter'])
    backend.restore_restart_state(fresh)
    return dict(fields=receipt,land_state=land_receipt,step_index=int(step),time_seconds=payload['time_seconds'],
                metadata_note='backend call counters retained from coarsest owner class; cadence index rebuilt')


def remap_step_carriers(payload, weights, layouts):
    """Carry physics-to-dynamics feedback without reinitializing accumulators."""
    previous=[]
    if any(item is None for item in payload['previous']):
        if not all(item is None for item in payload['previous']):
            raise ValueError('partial previous-surface carrier')
        previous=[None]*len(layouts)
    else:
        keys=set(payload['previous'][0])
        global_values={key: weights.intensive(gather_owner_columns(
            [item[key] for item in payload['previous']],payload['layouts'],len(weights.source_area))[0]).astype(
                payload['previous'][0][key].dtype) for key in keys}
        previous=[{key: np.ascontiguousarray(value[...,layout.cell_l2g])
                   for key,value in global_values.items()} for layout in layouts]
    forcing=[]
    if any(item is None for item in payload['forcing']):
        if not all(item is None for item in payload['forcing']):
            raise ValueError('partial GF feedback carrier')
        forcing=[None]*len(layouts)
    else:
        global_values={key: atmospheric_intensive(gather_owner_columns(
            [item[key] for item in payload['forcing']],payload['layouts'],len(weights.source_area))[0], weights).astype(
                payload['forcing'][0][key].dtype) for key in ('rthdynten','rqvdynten')}
        forcing=[dict(time_seconds=payload['forcing'][0]['time_seconds'],
                      **{key:np.ascontiguousarray(value[...,layout.cell_l2g])
                         for key,value in global_values.items()}) for layout in layouts]
    return dict(previous=previous,forcing=forcing)
