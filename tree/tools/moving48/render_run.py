"""Seamless products for a moving-region run: one fixed render lattice, continuous loops, a track chart.

  render_run.py RUN_DIR OUT_DIR [--products P1,P2] [--window lambert:...] [--margin-km 500]
                [--dx-m 3000] [--best-track obs/best-track.json] [--jobs 4]

Every history frame of the run, whatever member mesh it came from, is resampled by the
Rust converter onto ONE Lambert lattice chosen once for the whole run, and drawn by the
Rust renderer (``hexcore.cli render`` -> rw_mpas_convert -> rw_wrfbatch).  The lattice is
the bounding box of the run's own tracked centres and region centres plus a margin, at the
fine spacing, unless ``--window`` names one.  A moving region therefore never changes the
map: frame N and frame N+1 share every pixel's geography, and a move shows only as what
the model did.

Loops (animated GIF, and MP4 when ffmpeg exists) are assembled from the Rust PNGs in lead
order; nothing is redrawn.  The track chart is an analysis chart (matplotlib, coastlines
from the renderer's Natural Earth basemap): model track, region centres with their fine
interiors at each move, and the best track when given for scoring.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

EARTH_KM = 6371.229
DEFAULT_PRODUCTS = 'composite_reflectivity,10m_wind_speed_and_direction,mslp_10m_winds'


def km_per_deg_lon(lat):
    return math.pi * EARTH_KM / 180.0 * math.cos(math.radians(lat))


def choose_window(points, margin_km, dx_m):
    lats = [p[0] for p in points]
    lons = [p[1] % 360 for p in points]
    lat0 = (min(lats) + max(lats)) / 2
    lon0 = (min(lons) + max(lons)) / 2
    height_km = (max(lats) - min(lats)) * math.pi * EARTH_KM / 180 + 2 * margin_km
    width_km = (max(lons) - min(lons)) * km_per_deg_lon(lat0) + 2 * margin_km
    nx = int(math.ceil(width_km * 1000 / dx_m))
    ny = int(math.ceil(height_km * 1000 / dx_m))
    lon0 = lon0 - 360 if lon0 > 180 else lon0
    true = round(lat0, 2)
    return f'lambert:{lat0:.3f},{lon0:.3f},{int(dx_m)},{nx},{ny},{true},{true},{lon0:.3f}', dict(
        center=[lat0, lon0], dx_m=dx_m, nx=nx, ny=ny, width_km=width_km, height_km=height_km)


def read_polylines(path, bbox):
    """Plain ESRI shapefile PolyLine/Polygon reader (types 3 and 5), clipped by bounding box."""
    west, east, south, north = bbox
    lines = []
    data = Path(path).read_bytes()
    position = 100
    while position + 8 <= len(data):
        length = struct.unpack('>i', data[position + 4:position + 8])[0] * 2
        record = data[position + 8:position + 8 + length]
        position += 8 + length
        kind = struct.unpack('<i', record[:4])[0]
        if kind not in (3, 5):
            continue
        xmin, ymin, xmax, ymax = struct.unpack('<4d', record[4:36])
        if xmax < west or xmin > east or ymax < south or ymin > north:
            continue
        parts_n, points_n = struct.unpack('<2i', record[36:44])
        parts = list(struct.unpack(f'<{parts_n}i', record[44:44 + 4 * parts_n])) + [points_n]
        base = 44 + 4 * parts_n
        coords = struct.unpack(f'<{2 * points_n}d', record[base:base + 16 * points_n])
        for a, b in zip(parts[:-1], parts[1:]):
            lines.append((coords[2 * a:2 * b:2], coords[2 * a + 1:2 * b:2]))
    return lines


def circle(lat, lon, radius_km, n=120):
    out_lat, out_lon = [], []
    for k in range(n + 1):
        bearing = 2 * math.pi * k / n
        d = radius_km / EARTH_KM
        la1, lo1 = math.radians(lat), math.radians(lon)
        la2 = math.asin(math.sin(la1) * math.cos(d) + math.cos(la1) * math.sin(d) * math.cos(bearing))
        lo2 = lo1 + math.atan2(math.sin(bearing) * math.sin(d) * math.cos(la1),
                               math.cos(d) - math.sin(la1) * math.sin(la2))
        out_lat.append(math.degrees(la2))
        out_lon.append(math.degrees(lo2))
    return out_lat, out_lon


def region_centres(state):
    """(from model seconds, lat, lon) of every region the run used, from its move record."""
    first = state['frames'][0]['member_center'] if state['frames'] else None
    out = [] if first is None else [(0.0, first[0], first[1])]
    for move in state.get('moves', []):
        out.append((float(move['at_seconds']), move['target_center'][0], move['target_center'][1]))
    return out


def track_chart(state, interior_km, best, basemap, out_png, start_time, until=None, bbox=None, size=(11, 8)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    frames = state['frames']
    fixes = [(f['model_seconds'], f['fix']['lat'], f['fix']['lon']) for f in frames if f['fix'].get('ok')]
    centres = region_centres(state)
    if until is not None:
        fixes = [p for p in fixes if p[0] <= until]
        centres = [c for c in centres if c[0] <= until]
    lats = [p[1] for p in fixes] + [c[1] for c in centres]
    lons = [p[2] for p in fixes] + [c[2] for c in centres]
    t0 = datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S')
    t1 = t0 + timedelta(seconds=state['t'])
    if until is not None:
        t1 = t0 + timedelta(seconds=until)
    if best:
        rows = [r for r in best['rows'] if t0 <= datetime.strptime(r['time'], '%Y-%m-%d %H:%M:%S') <= t1]
        lats += [r['lat'] for r in rows]
        lons += [r['lon'] for r in rows]
    if bbox is None:
        pad = interior_km / 111.0 + 1.0
        bbox = (min(lons) - pad, max(lons) + pad, min(lats) - pad, max(lats) + pad)
    fig, ax = plt.subplots(figsize=size, dpi=100)
    shp = Path(basemap) / 'natural_earth_10m' / 'ne_10m_coastline.shp' if basemap else None
    if basemap and not shp.is_file():
        found = sorted(Path(basemap).rglob('*coastline*.shp'))
        shp = found[0] if found else None
    if shp:
        for x, y in read_polylines(shp, bbox):
            ax.plot(x, y, color='#6b6b6b', lw=0.7, zorder=1)
    for index, (seconds, lat, lon) in enumerate(centres):
        clat, clon = circle(lat, lon, interior_km)
        latest = index == len(centres) - 1
        ax.plot(clon, clat, color='#3b7dd8', lw=1.8 if latest else 0.8, alpha=1.0 if latest else 0.45, zorder=2,
                label='3 km interior of each region' if index == 0 else None)
        ax.plot(lon, lat, '+', color='#3b7dd8', ms=9, zorder=3)
        ax.annotate(f'region {index} from {seconds/3600:g} h', (lon, lat), textcoords='offset points',
                    xytext=(6, -12), fontsize=8, color='#3b7dd8')
    ax.plot([p[2] for p in fixes], [p[1] for p in fixes], '-o', color='#d8543b', ms=3, lw=1.6, zorder=4,
            label=f"model track ({state['fixes_source']})")
    for seconds, lat, lon in fixes:
        if seconds % 43200 == 0:
            ax.annotate(f'{seconds/3600:.0f} h', (lon, lat), textcoords='offset points', xytext=(5, 5), fontsize=8,
                        color='#d8543b')
    if best:
        ax.plot([r['lon'] for r in rows], [r['lat'] for r in rows], '-s', color='#202020', ms=3, lw=1.2, zorder=4,
                label=f"best track ({best.get('source', 'IBTrACS')}), scoring only")
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect(1.0 / math.cos(math.radians((bbox[2] + bbox[3]) / 2)))
    ax.set_xlabel('longitude (deg E)')
    ax.set_ylabel('latitude (deg N)')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    shown = state['t'] if until is None else until
    ax.set_title(f'Moving 3 km region: model track and regions, {start_time[:13]}Z + {shown/3600:g} h '
                 f'({max(len(centres) - 1, 0)} moves)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def lead_minutes(path):
    match = re.search(r'_f(\d+)(?:h(\d+)m)?\.png$', Path(path).name)
    return int(match.group(1)) * 60 + int(match.group(2) or 0)


def mslp_track_loop(out, state, interior_km, best, basemap, start_time):
    """Each Rust MSLP frame beside the track chart up to that frame's lead (one bounding box)."""
    from PIL import Image
    pngs = sorted(out.joinpath('frames').rglob('*_mslp_10m_winds_f*.png'), key=lead_minutes)
    if not pngs:
        return None
    fixes = [(f['fix']['lat'], f['fix']['lon']) for f in state['frames'] if f['fix'].get('ok')]
    centres = region_centres(state)
    lats = [p[0] for p in fixes] + [c[1] for c in centres]
    lons = [p[1] for p in fixes] + [c[2] for c in centres]
    pad = interior_km / 111.0 + 1.0
    bbox = (min(lons) - pad, max(lons) + pad, min(lats) - pad, max(lats) + pad)
    panels = out / 'mslp-track'
    panels.mkdir(exist_ok=True)
    frames = []
    for png in pngs:
        minutes = lead_minutes(png)
        chart = panels / f'track-{minutes:05d}.png'
        track_chart(state, interior_km, best, basemap, chart, start_time, until=minutes * 60.0, bbox=bbox,
                    size=(9, 9))
        left = Image.open(png).convert('RGB')
        right = Image.open(chart).convert('RGB')
        right = right.resize((int(right.width * left.height / right.height), left.height))
        both = Image.new('RGB', (left.width + right.width, left.height), 'white')
        both.paste(left, (0, 0))
        both.paste(right, (left.width, 0))
        path = panels / f'mslp-track-{minutes:05d}.png'
        both.save(path)
        frames.append(both)
    gif = out / 'loop_mslp_track.gif'
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=400, loop=0, optimize=True)
    return dict(gif=str(gif), frames=len(frames))


def assemble_loops(out, products):
    from PIL import Image
    loops = {}
    for product in products:
        pngs = sorted(out.joinpath('frames').rglob(f'*_{product}_f*.png'), key=lead_minutes)
        if not pngs:
            pngs = sorted((p for p in out.joinpath('frames').rglob('*.png') if f'/{product}/' in p.as_posix()),
                          key=lambda p: p.name)
        if not pngs:
            continue
        images = [Image.open(p).convert('RGB') for p in pngs]
        gif = out / f'loop_{product}.gif'
        images[0].save(gif, save_all=True, append_images=images[1:], duration=400, loop=0, optimize=True)
        loops[product] = dict(gif=str(gif), frames=len(pngs), first=str(pngs[0]), last=str(pngs[-1]))
        if shutil.which('ffmpeg'):
            listing = out / f'.{product}.txt'
            listing.write_text(''.join(f"file '{p}'\nduration 0.4\n" for p in pngs))
            mp4 = out / f'loop_{product}.mp4'
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing),
                            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-pix_fmt', 'yuv420p', str(mp4)], check=False)
            listing.unlink()
            if mp4.is_file():
                loops[product]['mp4'] = str(mp4)
    return loops


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--products', default=DEFAULT_PRODUCTS)
    p.add_argument('--window', help='explicit lambert:... lattice (default: from the run track)')
    p.add_argument('--margin-km', type=float, default=500.0)
    p.add_argument('--dx-m', type=float, default=3000.0)
    p.add_argument('--best-track', type=Path)
    p.add_argument('--basemap', type=Path, default=None,
                   help='Natural Earth basemap folder for the track chart (default $RUSTWX_BASEMAP_DIR)')
    p.add_argument('--label', default='hexmoving')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--every-hours', type=float, default=1.0, help='render frames on this cadence')
    p.add_argument('--size', default='1200x900')
    a = p.parse_args()
    state = json.loads((a.run / 'controller.json').read_text())
    case = json.loads(Path(state['case']).read_text())
    mesh = case['mesh']
    interior = float(mesh['fine_radius_km']) - float(mesh.get('fine_transition_km', 0.0))
    members = state['members']
    frames = {}
    for f in state['frames']:
        if f['model_seconds'] % (a.every_hours * 3600) == 0 and Path(f['history']).is_file():
            frames[f['model_seconds']] = f        # a recovered segment's frame replaces the lost one
    points = [(f['fix']['lat'], f['fix']['lon']) for f in state['frames'] if f['fix'].get('ok')]
    points += [tuple(f['member_center']) for f in state['frames']]
    if a.window:
        window, lattice = a.window, dict(explicit=True)
    else:
        window, lattice = choose_window(points, a.margin_km + interior * 0.0, a.dx_m)
    a.out.mkdir(parents=True, exist_ok=True)
    scratch = a.out.parent / (a.out.name + '.render-scratch')
    groups = {}
    for seconds, f in sorted(frames.items()):
        groups.setdefault((f['segment'], f['member']), []).append(f)
    jobs = []
    for (segment, name), rows in groups.items():
        member = members[name]
        # one render call per frame keeps each conversion small and lets frames run in parallel
        for f in rows:
            tag = f'{int(f["model_seconds"]):07d}'
            cmd = [sys.executable, '-m', 'hexcore.cli', 'render', '--history', f['history'],
                   '--mesh', member['static'], '--init', member['init'],
                   '--simulation-start', case['start_time'], '--out', str(a.out / 'frames'),
                   '--scratch', str(scratch / tag), '--model-label', 'gpuwm-hex moving 3 km',
                   '--window', window, '--view', 'track', '--label', a.label,
                   '--products', a.products, '--size', a.size]
            jobs.append((tag, cmd))
    logs = a.out / 'logs'
    logs.mkdir(exist_ok=True)

    def run(job):
        tag, cmd = job
        with (logs / f'render-{tag}.log').open('w') as log:
            return tag, subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode

    with futures.ThreadPoolExecutor(a.jobs) as pool:
        results = dict(pool.map(run, jobs))
    products = a.products.split(',')
    loops = assemble_loops(a.out, products)
    best = json.loads(a.best_track.read_text()) if a.best_track else None
    # The renderer's Natural Earth basemap; without one the track chart is drawn without coastlines.
    basemap = a.basemap or os.environ.get('RUSTWX_BASEMAP_DIR')
    track_chart(state, interior, best, basemap, a.out / 'track.png', case['start_time'])
    if 'mslp_10m_winds' in products:
        loops['mslp_track'] = mslp_track_loop(a.out, state, interior, best, basemap, case['start_time'])
    manifest = dict(run=str(a.run), window=window, lattice=lattice, frames=len(jobs),
                    failed={k: v for k, v in results.items() if v}, loops=loops, track_chart=str(a.out / 'track.png'),
                    renderer='hexcore.cli render -> rw_mpas_convert -> rw_wrfbatch (Rust); loops assembled from '
                             'those PNGs; track chart is an analysis chart')
    (a.out / 'products.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest, indent=1))
    if manifest['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
