"""Run the predictive tracker over forecast output (CPU only): hex history files or lat/lon GRIB forecasts.

usage:
  track_history.py --case CASE.json --out TRACK.json (--histories FILE... | --history-dir DIR | --grib FILE...)
                   [--start-time YYYY-mm-dd_HH:MM:SS] [--simulate-placement] [--check-minutes N]

The tracker reads only the forecast fields (MSLP, winds at pressure levels, 10 m wind, or an object field).
The case JSON supplies the analysis-time storm position (storm.initial_center_deg) and an optional "tracker"
block (TrackerConfig.from_dict); the placement interior comes from the case "mesh" block when present.

--simulate-placement replays the placement policy on these fixes, checking at each output time, and reports
the storm-to-region distance at every output (an offline A2 check; the forecast itself did not move).
Moving-run histories carry their own mesh per segment: the tracker rebuilds its point cloud whenever the
cell coordinates change.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

from hexcore import storm_tracker as st

HISTORY_TIME = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2})')


def history_time(path: Path) -> datetime:
    m = HISTORY_TIME.search(path.name)
    if not m:
        raise ValueError(f'no time stamp in {path}')
    return datetime.strptime(m.group(1), '%Y-%m-%d_%H.%M.%S').replace(tzinfo=timezone.utc)


class HistoryReader:
    """Reads the tracker's fields from a hex/MPAS history file; caches the point cloud per mesh."""

    def __init__(self, object_field: str | None = None, near_km: float = 900.0):
        self.cloud = None
        self.key = None
        self.object_field = object_field
        self.near_km = near_km

    def __call__(self, path: Path, guess) -> st.StateFields:
        import netCDF4
        with netCDF4.Dataset(path) as d:
            v = d.variables
            lat = np.asarray(v['latCell'][:], float)
            lon = np.asarray(v['lonCell'][:], float)
            if np.nanmax(np.abs(lat)) <= np.pi / 2 + 1e-6:
                lat, lon = np.rad2deg(lat), np.rad2deg(lon)
            key = (lat.size, float(lat[::997].sum()), float(lon[::997].sum()))
            if key != self.key:
                self.cloud, self.key = st.PointCloud(lat, lon), key
            ter = np.asarray(v['ter'][:], float).ravel()
            ps = np.asarray((v['surface_pressure'] if 'surface_pressure' in v else v['psfc'])[0], float)
            if 't2' in v:
                t_low = np.asarray(v['t2'][0], float)
            else:
                t_low = (np.asarray(v['theta'][0, :, 0], float)
                         * (np.asarray(v['pressure'][0, :, 0], float) / 1e5) ** (st.RD / st.CP))
            sub = self.cloud.near(guess[0], guess[1], self.near_km)
            pressure = u = w = None
            if 'pressure' in v and 'u_zonal' in v:
                # Read whole variables (contiguous) and slice: far faster than netCDF fancy indexing.
                pressure = np.asarray(v['pressure'][0], np.float32)[sub]
                u = np.asarray(v['u_zonal'][0], np.float32)[sub]
                w = np.asarray(v['v_meridional'][0], np.float32)[sub]
            obj = None
            if self.object_field:
                name = self.object_field
                raw = np.asarray(v[name][0], float)
                obj = raw.max(axis=-1) if raw.ndim == 2 else raw
            u10 = np.asarray(v['u10'][0], float) if 'u10' in v else None
            v10 = np.asarray(v['v10'][0], float) if 'v10' in v else None
        f = st.fields_from_columns(lat, lon, psfc=ps, terrain=ter, t_low=t_low, u10=u10, v10=v10,
                                   object_field=obj, cloud=self.cloud)
        if pressure is not None:
            n = lat.size
            f.u_p, f.v_p = {}, {}
            for level in (85000.0, 70000.0) + st.STEERING_LEVELS_PA[2:]:
                uu, vv = np.full(n, np.nan), np.full(n, np.nan)
                uu[sub] = st.interpolate_to_pressure(pressure, u, level)
                vv[sub] = st.interpolate_to_pressure(pressure, w, level)
                f.u_p[level], f.v_p[level] = uu, vv
        return f


def grib_fields(path: Path, box=None) -> st.StateFields:
    import eccodes
    got = {}
    with open(path, 'rb') as stream:
        while True:
            g = eccodes.codes_grib_new_from_file(stream)
            if g is None:
                break
            name = eccodes.codes_get(g, 'shortName')
            ni, nj = eccodes.codes_get(g, 'Ni'), eccodes.codes_get(g, 'Nj')
            values = eccodes.codes_get_values(g).reshape(nj, ni)
            la = np.linspace(eccodes.codes_get(g, 'latitudeOfFirstGridPointInDegrees'),
                             eccodes.codes_get(g, 'latitudeOfLastGridPointInDegrees'), nj)
            lo = eccodes.codes_get(g, 'longitudeOfFirstGridPointInDegrees') + np.arange(ni) * eccodes.codes_get(
                g, 'iDirectionIncrementInDegrees')
            eccodes.codes_release(g)
            LO, LA = np.meshgrid(lo, la)
            got[name] = values.ravel()
            got['lat'], got['lon'] = LA.ravel(), st.wrap_lon(LO.ravel())
    keep = np.ones(got['lat'].size, bool)
    if box is not None:
        lat0, lon0, radius = box
        keep = st.distance_km(lat0, lon0, got['lat'], got['lon']) <= radius
    p = got.get('prmsl', got.get('msl'))
    f = st.StateFields(lat=got['lat'][keep], lon=got['lon'][keep], mslp_pa=p[keep])
    if '10u' in got:
        f.wind10 = np.hypot(got['10u'], got['10v'])[keep]
    return f


def simulate_placement(session: st.TrackSession, rows, region0):
    """Replay the policy at every output time; region in effect at each output and the storm distance."""
    region = list(region0)
    out, moves = [], []
    for row in rows:
        t = row['seconds']
        fix = row['fix']
        before = region
        d_before = None if not fix['ok'] else float(st.distance_km(*before, fix['lat'], fix['lon']))
        decision = session.policy.decide(t, session.fixes[:row['index'] + 1], region, misses=row['misses'])
        if decision.move:
            region = decision.centre
            moves.append(dict(seconds=t, centre=region, moved_km=float(st.distance_km(*before, *region)),
                              reason=decision.reason))
        d_after = None if not fix['ok'] else float(st.distance_km(*region, fix['lat'], fix['lon']))
        out.append(dict(seconds=t, region_before=before, region_after=list(region), storm_offset_before_km=d_before,
                        storm_offset_after_km=d_after, decision=decision.as_dict()))
    return out, moves


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--histories', type=Path, nargs='*', default=[])
    p.add_argument('--history-dir', type=Path, action='append', default=[])
    p.add_argument('--grib', type=Path, nargs='*', default=[], help='files named ...-fHHH.grib2 (lead hours)')
    p.add_argument('--start-time')
    p.add_argument('--label', default='')
    p.add_argument('--simulate-placement', action='store_true')
    p.add_argument('--check-minutes', type=float, help='policy check period for the replay (default: output interval)')
    p.add_argument('--interior-radius-km', type=float, help='override the case mesh interior for the replay')
    p.add_argument('--max-move-km', type=float, help='policy override for the replay')
    p.add_argument('--core-margin-km', type=float, help='policy override for the replay')
    p.add_argument('--follow', choices=('lazy', 'continuous'), help='policy override for the replay')
    p.add_argument('--min-move-km', type=float, help='policy override for the replay')
    p.add_argument('--hours', type=float, default=1e9)
    a = p.parse_args(argv)
    case = json.loads(a.case.read_text())
    start = datetime.strptime(a.start_time or case['start_time'], '%Y-%m-%d_%H:%M:%S').replace(tzinfo=timezone.utc)
    tcfg = dict(case.get('tracker', {}))
    tcfg.setdefault('initial_center_deg', case['storm']['initial_center_deg'])
    tcfg.pop('placement', None)  # built below, with the interior radius from the case mesh
    config = st.TrackerConfig.from_dict(tcfg)
    if a.simulate_placement:
        place = dict(case.get('tracker', {}).get('placement', {}))
        if a.interior_radius_km:
            place['interior_radius_km'] = a.interior_radius_km
        place.setdefault('interior_radius_km', st.interior_radius_from_mesh(case['mesh']))
        if a.max_move_km is not None:
            place['max_move_km'] = a.max_move_km
        if a.core_margin_km is not None:
            place['core_margin_km'] = a.core_margin_km
        if a.follow is not None:
            place['follow'] = a.follow
        if a.min_move_km is not None:
            place['min_move_km'] = a.min_move_km
        config.placement = st.PlacementConfig(**place)
    items = []
    files = list(a.histories)
    for folder in a.history_dir:
        files += sorted(folder.glob('*history*.nc'))
    for f in files:
        items.append(((history_time(f) - start).total_seconds(), 'history', f))
    for f in a.grib:
        m = re.search(r'-f(\d{3})\.grib2$', f.name)
        items.append((int(m.group(1)) * 3600.0, 'grib', f))
    items = sorted(i for i in items if 0 <= i[0] <= a.hours * 3600 + 1e-6)
    seen = set()
    items = [i for i in items if not (i[0] in seen or seen.add(i[0]))]
    if not items:
        sys.exit('no forecast output to track')
    gaps = np.diff([i[0] for i in items])
    if a.simulate_placement and config.placement is not None:
        config.placement.check_minutes = a.check_minutes or float(np.median(gaps) / 60.0 if gaps.size else 60.0)
    session = st.TrackSession(config)
    reader = HistoryReader(object_field=(config.object.field_name if config.mode == 'object' else None))
    rows = []
    for seconds, kind, path in items:
        guess, _ = session.predicted(seconds)
        started = time.monotonic()
        fields = reader(path, guess) if kind == 'history' else grib_fields(path, (guess[0], guess[1], 1500.0))
        fix = session.observe(seconds, fields)
        rows.append(dict(seconds=seconds, lead_h=seconds / 3600.0, file=str(path), index=len(session.fixes) - 1,
                         misses=session.misses, fix=fix.as_dict(), tracker_seconds=time.monotonic() - started))
        status = f'{fix.lat:7.3f} {fix.lon:8.3f} p={fix.mslp_min_hpa or float("nan"):7.1f}' if fix.ok else 'MISS ' + fix.reason
        print(f'{a.label} +{seconds / 3600:5.1f} h  {status}  ({rows[-1]["tracker_seconds"]:.1f} s)', flush=True)
    result = dict(label=a.label, case=case.get('case'), start_time=start.strftime('%Y-%m-%d_%H:%M:%S'),
                  tracker_config=config.as_dict(), track=rows, lost=session.lost)
    if a.simulate_placement:
        replay, moves = simulate_placement(session, rows, config.initial_center_deg)
        offsets = [r['storm_offset_before_km'] for r in replay if r['storm_offset_before_km'] is not None]
        offsets += [r['storm_offset_after_km'] for r in replay if r['storm_offset_after_km'] is not None]
        interior = config.placement.interior_radius_km
        result['placement_replay'] = dict(
            check_minutes=config.placement.check_minutes, interior_radius_km=interior,
            safe_radius_km=config.placement.safe_km(), moves=moves, rows=replay,
            max_storm_offset_km=max(offsets) if offsets else None,
            a2_inside_interior_every_output=bool(offsets) and max(offsets) <= interior,
            core_inside_safe_every_output=bool(offsets) and max(offsets) <= config.placement.safe_km())
        worst = result['placement_replay']['max_storm_offset_km']
        print(f'{a.label} replay: {len(moves)} moves, max storm offset '
              f'{"none (no fix)" if worst is None else f"{worst:.1f} km"} (interior {interior:.0f}, '
              f'safe {config.placement.safe_km():.0f}); lost={session.lost}')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
