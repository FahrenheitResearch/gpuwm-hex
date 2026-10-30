"""Strict conversion of known soil hydraulics, independent of data source.

The middle branch preserves availability relative to wilting point and field
capacity, as in HRRR soil ingestion. Dry and wet branches preserve position
between zero/wilt and field capacity/saturation, so a valid source state maps
inside target capacity without clipping. This function needs actual physical
anchors. Interpolated categorical class numbers cannot establish those anchors.
"""
import numpy as np


def remap_relative_wetness(values, source_knots, target_knots):
    """Map (cells,layers) using per-cell [zero,wilt,reference,saturation].

    Refuse invalid input or anchors before returning any output. Callers must
    supply soil columns only; water/ice reservoirs have a separate contract.
    Liquid water is handled by retaining the source liquid/total water fraction
    after converting the total water, not by independently moving wilt points.
    """
    original = np.asarray(values)
    a = np.asarray(values, dtype=np.float64)
    src = np.asarray(source_knots, dtype=np.float64)
    dst = np.asarray(target_knots, dtype=np.float64)
    if a.ndim != 2 or src.shape != (a.shape[0], 4) or dst.shape != src.shape:
        raise ValueError('soil values/anchors shape must be (cells,layers)/(cells,4)')
    for label, knots in [('source', src), ('target', dst)]:
        if (not np.isfinite(knots).all() or np.any(knots[:, 0] != 0)
                or np.any(np.diff(knots, axis=1) <= 0) or np.any(knots[:, -1] > 1)):
            raise ValueError(f'{label} anchors must satisfy 0 < wilt < reference < saturation <= 1')
    if not np.isfinite(a).all() or np.any(a < 0) or np.any(a > src[:, -1, None]):
        raise ValueError('soil moisture outside finite source bounds; refusing remap')
    segment = (a >= src[:, 1, None]).astype(int) + (a >= src[:, 2, None])
    row = np.arange(len(a))[:, None]
    lo, hi = src[row, segment], src[row, segment + 1]
    fraction = (a - lo) / (hi - lo)
    dlo, dhi = dst[row, segment], dst[row, segment + 1]
    mapped = dlo * (1 - fraction) + dhi * fraction
    same = np.all(src == dst, axis=1)
    mapped[same] = a[same]
    dtype = original.dtype if np.issubdtype(original.dtype, np.floating) else np.dtype('float64')
    mapped = mapped.astype(dtype)
    # Target storage arithmetic is part of the contract, as in land_state.
    if not np.isfinite(mapped).all() or np.any(mapped < 0) or np.any(mapped > dst[:, -1, None].astype(dtype)):
        raise ValueError('mapped soil moisture outside target storage bounds')
    return mapped, dict(schema='hex.soil-relative-wetness/v1',
                        method='piecewise-zero-wilt-reference-saturation',
                        cells=len(a), values=a.size,
                        changed_values=int(np.count_nonzero(mapped != original)),
                        clipped_values=0,
                        max_abs_change=float(np.max(np.abs(mapped.astype(float) - a), initial=0)))


def knots_from_classes(classes, table):
    """Select physical anchors from integral categorical labels, without repair."""
    classes = np.asarray(classes)
    table = np.asarray(table)
    if classes.ndim != 1 or table.ndim != 2 or table.shape[1] != 4:
        raise ValueError('class/table shape must be (cells,)/(classes,4)')
    if (not np.isfinite(classes).all() or np.any(classes != np.floor(classes))
            or np.any(classes < 0) or np.any(classes >= len(table))):
        raise ValueError('source soil classes must be integral, in-range categorical labels; refusing rounding')
    return table[classes.astype(int)]


def donor_quality(moisture, liquid, capacity):
    """Reject invalid donor columns, never change their values or the DA gate."""
    sm, sw, cap = map(np.asarray, (moisture, liquid, capacity))
    if sm.ndim != 2 or sw.shape != sm.shape or cap.shape != sm.shape[:1]:
        raise ValueError('donor shape mismatch')
    bad = (~np.isfinite(sm) | ~np.isfinite(sw) | (sw < 0) | (sw > sm)
           | (sm > cap[:, None]) | ~np.isfinite(cap[:, None]) | (cap[:, None] <= 0))
    good = ~np.any(bad, axis=1)
    if not good.any():
        raise ValueError('no physical soil donor columns remain')
    return good, dict(source_rejected_columns=int((~good).sum()),
                      source_rejected_values=int(bad.sum()),
                      source_accepted_columns=int(good.sum()))


def source_contract(map_sources, name=None):
    """Select policy by table metadata, with no source-specific execution path."""
    import json
    from pathlib import Path
    rows = json.loads((Path(__file__).resolve().parents[1] / 'data/soil-hydraulics.v1.json').read_text())['contracts']
    matches = [r for r in rows if (r['id'] == name if name else
               r.get('default_enabled', False) and set(map_sources) & set(r['map_sources']))]
    if len(matches) > 1 or (name and not matches):
        raise ValueError('soil hydraulics contract is missing or ambiguous')
    return matches[0] if matches else None


def remap_cold_init(path, donor, contract, start_time):
    """Admit a co-located native donor, then remap valid soil onto cold init.

    The source pgrb2 categorical field can be spatially interpolated. Use the
    native surface NetCDF with categorical classes, co-located total/liquid
    moisture and a declared hydraulic table. A rejected donor is excluded from
    the spatial query and counted. Every target soil cell must find a valid
    donor within the table's distance bound. No state is clipped.
    """
    import hashlib
    from datetime import datetime
    from pathlib import Path
    from netCDF4 import Dataset, num2date
    from scipy.spatial import cKDTree
    from .land_state import context, read, porosity, validate_land, LAND_ATTRIBUTE
    from hexcore.mpasphys.noah_v842 import LANDUSE_URBAN_INDEX
    spec = contract['source']; names = spec['fields']
    stamp = datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S')
    if not contract['valid_from'] <= stamp.strftime('%Y-%m-%d') <= contract['valid_through']:
        raise ValueError('soil contract does not cover this analysis date')
    with Dataset(path) as target:
        if LAND_ATTRIBUTE in target.ncattrs():
            raise ValueError('cold soil remap refuses a carried warm background')
        ctx = context(target)
        sm = read(target['smois']); sw = read(target['sh2o']); st = read(target['tslb'])
        if sm.shape[1] != len(contract['target']['soil_layer_depths_m']):
            raise ValueError('soil remap target layer geometry mismatch')
        if not np.allclose(read(target['dzs']), contract['target']['soil_layer_depths_m'], rtol=0, atol=1e-7):
            raise ValueError('soil remap target layer depths mismatch')
        cap = porosity(ctx, len(sm)); target_ids = np.flatnonzero(cap < 1)
        lat = read(target['latCell']); lon = read(target['lonCell'])
        before_violations = int(np.count_nonzero((sm > cap[:, None]) | (sw > sm) | (sw < 0)))
    soil = np.asarray(ctx['isltyp'], dtype=int).copy()
    soil[(soil == 14) & (np.asarray(ctx['xice']) == 0)] = 7
    target_knots = knots_from_classes(soil[target_ids], np.array(contract['target']['soil_knots'], dtype='f4'))
    lu = np.asarray(ctx['mminlu']).tobytes().decode().strip(' \x00')
    urban = np.asarray(ctx['ivgtyp'])[target_ids] == LANDUSE_URBAN_INDEX[lu]
    target_knots[urban] = np.asarray(contract['target']['urban_knots'], dtype='f4')
    if not np.array_equal(target_knots[:, -1], cap[target_ids]):
        raise ValueError('target hydraulic table disagrees with live Noah porosity')
    with Dataset(donor) as ds:
        ds.set_auto_mask(False)
        t = ds['time']; dt = num2date(t[0], t.units, only_use_cftime_datetimes=False)
        if dt.strftime('%Y-%m-%d_%H:%M:%S') != start_time:
            raise ValueError('soil donor time differs from initialization')
        land = np.asarray(ds[names['land']][0])
        veg = np.asarray(ds[names['vegetation_class']][0])
        eligible = (land == spec['land_value']) & (veg != spec['ice_vegetation_class'])
        ij = np.argwhere(eligible)
        classes = np.asarray(ds[names['soil_class']][0])[eligible]
        knots = knots_from_classes(classes, np.array(spec['soil_knots'], dtype='f4'))
        gv = np.asarray(ds[names['vegetation_fraction']][0])[eligible] * spec['vegetation_fraction_scale']
        if not np.isfinite(gv).all() or np.any((gv < 0) | (gv > 1)):
            raise ValueError('source vegetation fraction is outside declared units')
        urban = veg[eligible] == spec['urban_class']
        if spec['urban_vegetation_blend']:
            knots[urban] = (np.array(spec['urban_knots'], dtype='f4')[None, :] * (1-gv[urban,None])
                            + knots[urban] * gv[urban,None])
        else:
            knots[urban] = spec['urban_knots']
        source_sm = np.stack([np.asarray(ds[n][0])[eligible] for n in names['moisture']], axis=1)
        source_sw = np.stack([np.asarray(ds[n][0])[eligible] for n in names['liquid']], axis=1)
        if source_sm.shape[1] != sm.shape[1] or spec['soil_layer_depths_m'] != contract['target']['soil_layer_depths_m']:
            raise ValueError('source and target soil layer geometry differ')
        good, qc = donor_quality(source_sm, source_sw, knots[:, -1])
        slat = np.deg2rad(np.asarray(ds[names['latitude']][:])[ij[:,0]])
        slon = np.deg2rad(np.asarray(ds[names['longitude']][:])[ij[:,1]])
    def xyz(la, lo):
        return np.column_stack((np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)))
    ids = np.flatnonzero(good)
    distances, nearest = cKDTree(xyz(slat[good], slon[good])).query(xyz(lat[target_ids], lon[target_ids]), workers=-1)
    nearest = ids[nearest]
    km = 2*6371.229*np.arcsin(distances/2)
    # THE BREAKAGE THIS PREVENTS (shape1-mesh, an RTX 4090 host, 2026-10-29): a land cell the static resolves but the donor's
    # land mask does not (Kerguelen, Tahiti: 3,362 and 3,403 km from the nearest admitted GFS soil column) refused the
    # WHOLE init, so a 20 km uniform globe and a 2 km zoom globe over Tahiti could not start from GFS at all. Soil
    # from 3,000 km away is still never used: such a cell keeps the cold init's own soil (what every init carried
    # before this remap), the receipt counts and places it, and the land gate below still checks it.
    reached = np.isfinite(km) & (km <= contract['max_donor_distance_km'])
    if not reached.any():
        raise ValueError(f'no admitted soil donor within {contract["max_donor_distance_km"]} km; max={km.max()}')
    converted, result = remap_relative_wetness(source_sm[nearest[reached]], knots[nearest[reached]],
                                               target_knots[reached])
    fraction = np.divide(source_sw[nearest[reached]], source_sm[nearest[reached]], out=np.zeros_like(converted),
                         where=source_sm[nearest[reached]] != 0)
    new_sm, new_sw = sm.copy(), sw.copy()
    new_sm[target_ids[reached]] = converted
    new_sw[target_ids[reached]] = converted * fraction
    kept = target_ids[~reached]
    # A kept cell holds the native cold-start soil, which can exceed its own class's porosity: the cold init carries
    # the source model's moisture in the SOURCE's soil classes (3,520 such cells in a 20 km GFS cold init, 2026-10-28
    # 12Z). The remap converts every reached cell into the target classes; a kept cell stays exactly as native MPAS-A
    # starts it, and Noah's SSTEP caps SMC at SMCMAX on the first step, draining the excess as runoff
    # (module_sf_noahlsm.F:4018-4030, hexcore/mpasphys/noah_v842.py). So the gate below checks every cell this remap
    # writes or vouches for, and counts, rather than refuses, the untouched kept cells.
    # THE BREAKAGE THIS PREVENTS (shape1-mesh, V2X6000, 2026-10-29): the 20 km GFS globe still refused after abf837a3,
    # "soil bounds cell=522193 layer=0: sh2o=0.0 smois=0.468 porosity=0.434" -- a Kerguelen cell (-49.1, 70.4) whose
    # cold soil is GFS's saturated 0.468 on a sandy-loam target class, a cell this remap never touched.
    kept_violations = int(np.count_nonzero((sm[kept] > cap[kept, None]) | (sw[kept] > sm[kept]) | (sw[kept] < 0)))
    unreached = {'cells': int(kept.size),
                 'max_donor_distance_km': float(km[~reached].max(initial=0)) if kept.size else 0.0,
                 'kept': 'cold init soil (native MPAS-A cold start)',
                 'bound_violations': kept_violations,
                 'cells_latlon_deg': [[round(float(np.degrees(lat[i])), 3), round(float(np.degrees(lon[i])), 3)]
                                      for i in kept[:50]]}
    # The same gate as every DA boundary, before modifying the cold file, on every cell but the untouched kept ones.
    checked = np.ones(len(sm), dtype=bool); checked[kept] = False
    per_cell = {k: (np.asarray(v)[checked] if np.ndim(v) == 1 and np.shape(v)[0] == len(sm) else v)
                for k, v in ctx.items()}
    validate_land(dict(smois=new_sm[checked], sh2o=new_sw[checked], tslb=st[checked]), per_cell)
    with open(donor, 'rb') as stream:
        donor_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    import json
    contract_digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    receipt = {**result, **qc, 'contract': contract['id'], 'contract_sha256': contract_digest,
               'donor': str(Path(donor).resolve()), 'donor_sha256': donor_digest,
               'source_valid_time': start_time, 'spatial_method': 'nearest-valid-soil-on-sphere',
               'maximum_donor_distance_km': float(km[reached].max(initial=0)),
               'cells_beyond_donor_bound': unreached,
               'cold_before_bound_violations': before_violations,
               'cold_after_bound_violations': kept_violations,
               'target_changed_moisture_values': int(np.count_nonzero(new_sm != sm)),
               'target_changed_liquid_values': int(np.count_nonzero(new_sw != sw)),
               'water_ice_columns_unchanged': int(len(sm)-len(target_ids))}
    with Dataset(path, 'r+') as target:
        for name, values in [('smois', new_sm), ('sh2o', new_sw)]:
            var = target[name]
            if var.dimensions[:1] == ('Time',): var[0] = values
            else: var[:] = values
    # A sidecar avoids rewriting a multi-GB classic NetCDF just to extend its
    # header. The init provenance also embeds this receipt before hashing output.
    receipt_path = Path(str(path) + '.soil-remap.json')
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt
