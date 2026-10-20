"""Fast spherical rotation of an existing global refinement template.

Topology and metric lengths are invariant under the rotation. Geography and
initial/reference-coordinate preparation must be rebuilt for the new position.
"""
from pathlib import Path
import hashlib
import json
import shutil
import time
import uuid
import numpy as np
from .moving_remap import sphere_xyz


def rotation_between(start, end):
    a, b = sphere_xyz(*np.deg2rad(start)), sphere_xyz(*np.deg2rad(end))
    c = float(a @ b)
    if c < -1 + 1e-10:
        raise ValueError('antipodal patch centers have no unique minimal rotation')
    v = np.cross(a, b)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1+c)


def rotate_grid(source, output, *, start, end):
    from netCDF4 import Dataset
    began = time.perf_counter()
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError('rotated mesh needs a new output path')
    rotation = rotation_between(start, end)
    shutil.copyfile(source, output)
    changed=[]
    with Dataset(output, 'r+') as ds:
        old_normal = None
        if 'angleEdge' in ds.variables:
            lat, lon = np.asarray(ds['latEdge'][:]), np.asarray(ds['lonEdge'][:])
            angle = np.asarray(ds['angleEdge'][:])
            east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
            north = np.stack((-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)), axis=-1)
            old_normal = (np.cos(angle)[:,None]*east + np.sin(angle)[:,None]*north) @ rotation.T
        for suffix in ('Cell', 'Edge', 'Vertex'):
            keys = [axis+suffix for axis in 'xyz']
            vector = np.stack([np.asarray(ds[key][:]) for key in keys], axis=-1) @ rotation.T
            for index, key in enumerate(keys):
                ds[key][:] = vector[:, index]; changed.append(key)
            unit = vector/np.linalg.norm(vector, axis=-1, keepdims=True)
            lat, lon = np.arcsin(np.clip(unit[:,2],-1,1)), np.arctan2(unit[:,1],unit[:,0])
            ds['lat'+suffix][:]=lat; ds['lon'+suffix][:]=lon
            changed.extend(['lat'+suffix, 'lon'+suffix])
            f = 'f'+suffix
            if f in ds.variables:
                ds[f][:] = 2*7.29212e-5*np.sin(lat); changed.append(f)
        for key in ('edgeNormalVectors', 'edgeTangentVectors', 'coeffs_reconstruct'):
            if key in ds.variables:
                value = np.asarray(ds[key][:])
                if value.shape[-1] != 3:
                    raise ValueError(f'{key} is not an earth Cartesian vector')
                ds[key][:] = value @ rotation.T; changed.append(key)
        if 'angleEdge' in ds.variables:
            lat, lon = np.asarray(ds['latEdge'][:]), np.asarray(ds['lonEdge'][:])
            east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
            north = np.stack((-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)), axis=-1)
            normal = old_normal
            ds['angleEdge'][:] = np.arctan2(np.einsum('ij,ij->i',normal,north),
                                          np.einsum('ij,ij->i',normal,east))
            changed.append('angleEdge')
        ds.setncattr('file_id', uuid.uuid4().hex)
        ds.setncattr('moving_patch_rotation', json.dumps(dict(start=start,end=end)))
    digest=hashlib.sha256()
    with output.open('rb') as stream:
        for block in iter(lambda: stream.read(8<<20), b''):
            digest.update(block)
    return dict(method='rigid global spherical refinement-template rotation',
                source=str(source), output=str(output), start=list(start), end=list(end),
                sha256=digest.hexdigest(), changed_variables=changed,
                wall_seconds=time.perf_counter()-began, geography_prepared=False)
