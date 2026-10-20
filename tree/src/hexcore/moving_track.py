"""Spherical storm-centroid tracking and externally supplied forecast tracks.

Uses the regional tracker's threshold/centroid, dead-band and accepted-move
semantics. Hex cells vote by area. Diagnostic sampling is independent of output.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .moving_remap import sphere_xyz

EARTH_RADIUS_M = 6371229.0


def position(xyz):
    xyz = np.asarray(xyz, dtype=float)
    xyz = xyz / np.linalg.norm(xyz)
    return float(np.rad2deg(np.arcsin(xyz[2]))), float(np.rad2deg(np.arctan2(xyz[1], xyz[0])))


def external_position(track, seconds):
    a = np.asarray(track, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 2 or not np.all(np.isfinite(a)):
        raise ValueError('track must contain finite [seconds,latitude,longitude] rows')
    if np.any(np.diff(a[:,0]) <= 0) or np.any(abs(a[:,1]) > 90) or not a[0,0] <= seconds <= a[-1,0]:
        raise ValueError('track must be ordered and cover the requested prediction time')
    i = min(np.searchsorted(a[:,0], seconds, side='right') - 1, len(a)-2)
    f = (seconds-a[i,0]) / (a[i+1,0]-a[i,0])
    p, q = sphere_xyz(np.deg2rad(a[i:i+2,1]), np.deg2rad(a[i:i+2,2]))
    angle = np.arccos(np.clip(p @ q, -1, 1))
    if angle > np.pi - 1e-6:
        raise ValueError('antipodal track segment has no unique interpolation')
    xyz = (1-f)*p + f*q if angle < 1e-8 else (np.sin((1-f)*angle)*p + np.sin(f*angle)*q)/np.sin(angle)
    return position(xyz)


@dataclass
class StormTracker:
    threshold_dbz: float = 35.0
    search_radius_km: float = 500.0
    deadband_km: float = 30.0
    smoothing: float = 0.5
    smoothed: object = None

    def diagnose(self, mesh, reflectivity, center, *, interval_seconds, steering_ms=(0.,0.)):
        lat = np.asarray(mesh.arrays['latCell']); lon = np.asarray(mesh.arrays['lonCell'])
        xyz = sphere_xyz(lat, lon)
        anchor = sphere_xyz(*np.deg2rad(center))
        field = np.asarray(reflectivity, dtype=float)
        if field.ndim == 2:
            field = field.max(axis=0)
        if field.shape != lat.shape or not np.all(np.isfinite(field)):
            raise ValueError('reflectivity must be a finite cell or level/cell diagnostic')
        active = (field >= self.threshold_dbz) & (xyz @ anchor >= np.cos(self.search_radius_km*1000/EARTH_RADIUS_M))
        ids = np.flatnonzero(active)
        if not len(ids):
            return dict(move=False, center=list(center), reason='no cluster above threshold')
        # Choose a connected storm, not the centroid between unrelated systems.
        cells = np.asarray(mesh.arrays['cellsOnCell'], dtype=int)
        if np.any(cells < -1) or np.any(cells >= len(lat)):
            raise ValueError('tracker requires zero-based mesh connectivity')
        lookup = np.full(len(lat), -1); lookup[ids] = np.arange(len(ids))
        row, col = [], []
        for local, cell in enumerate(ids):
            neighbors = cells[cell, :int(mesh.arrays['nEdgesOnCell'][cell])]
            neighbors = neighbors[neighbors >= 0]
            adjacent = lookup[neighbors]; adjacent = adjacent[adjacent >= 0]
            row.extend([local]*len(adjacent)); col.extend(adjacent.tolist())
        _, labels = connected_components(csr_matrix((np.ones(len(row)), (row,col)), shape=(len(ids),len(ids))), directed=False)
        weights = np.asarray(mesh.arrays['areaCell'])[ids] * (field[ids]-self.threshold_dbz+1)
        winning = int(np.argmax(np.bincount(labels, weights=weights)))
        keep = labels == winning
        centroid = np.sum(xyz[ids[keep]] * weights[keep,None], axis=0)
        centroid /= np.linalg.norm(centroid)
        if not 0 < self.smoothing <= 1:
            raise ValueError('smoothing must be in (0,1]')
        if self.smoothed is not None:
            centroid = self.smoothing*centroid+(1-self.smoothing)*self.smoothed
            centroid /= np.linalg.norm(centroid)
        self.smoothed = centroid.copy()
        clat, clon = np.deg2rad(position(centroid))
        east = np.array([-np.sin(clon), np.cos(clon), 0.])
        north = np.array([-np.sin(clat)*np.cos(clon), -np.sin(clat)*np.sin(clon), np.cos(clat)])
        velocity = float(steering_ms[0])*east+float(steering_ms[1])*north
        speed = np.linalg.norm(velocity)
        angle = speed*interval_seconds/EARTH_RADIUS_M
        predicted = centroid if speed == 0 else np.cos(angle)*centroid + np.sin(angle)*velocity/speed
        displacement = np.arccos(np.clip(predicted @ anchor, -1,1))*EARTH_RADIUS_M/1000
        return dict(move=bool(displacement >= self.deadband_km), center=list(position(predicted)),
                    centroid=list(position(centroid)), displacement_km=float(displacement),
                    cluster_cells=int(keep.sum()), cluster_area_m2=float(np.sum(np.asarray(mesh.arrays['areaCell'])[ids[keep]])),
                    steering_ms=list(steering_ms), prediction_seconds=float(interval_seconds))
