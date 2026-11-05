import csv,collections,json,hashlib,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
parser=argparse.ArgumentParser(description='Observation-only IBTrACS counts and wind chart; no model skill claim')
parser.add_argument('--csv',type=Path,required=True)
parser.add_argument('--year',type=int,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--minimum-wind-kt',type=float,required=True)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
start=time.perf_counter()
source=args.csv
points=[]
with source.open() as f:
    for r in csv.DictReader(f):
        if r['SEASON'].strip()!=str(args.year) or r['TRACK_TYPE'].strip()!='main': continue
        if r['BASIN'].strip()!='NA' or r['USA_STATUS'].strip() not in ('TS','HU','SS'): continue
        try: wind=float(r['USA_WIND'])
        except ValueError: continue
        if wind<args.minimum_wind_kt: continue
        t=datetime.fromisoformat(r['ISO_TIME'].strip())
        if t.hour%6 or t.minute: continue
        points.append(dict(sid=r['SID'],time=t.isoformat(),wind_kt=wind,
                           latitude=float(r['LAT']),longitude=float(r['LON'])))
tracks=collections.defaultdict(list)
for p in points:tracks[p['sid']].append(p)
monthly=collections.Counter(datetime.fromisoformat(min(p['time'] for p in pts)).month for pts in tracks.values())
daily={}
for p in points:daily[p['time'][:10]]=max(daily.get(p['time'][:10],0),p['wind_kt'])
fig,axes=plt.subplots(2,1,figsize=(9,6),layout='constrained')
axes[0].bar(range(1,13),[monthly[m] for m in range(1,13)],color='#246e98')
axes[0].set(xticks=range(1,13),xlabel=f'Month of first 6-hour TS/SS/HU point with wind ≥{args.minimum_wind_kt:g} kt',ylabel='Systems',title=f'{args.year} North Atlantic: {len(tracks)} observed systems')
# Points avoid a misleading continuous line across days with no named storm.
axes[1].scatter([datetime.fromisoformat(k) for k in sorted(daily)],[daily[k] for k in sorted(daily)],s=9,color='#246e98')
axes[1].set(xlabel=f'Date in {args.year}',ylabel='Maximum observed wind (kt)',title='Daily maximum of available 6-hour NHC winds')
fig.suptitle('IBTrACS observations only. No model run or skill claim.',fontsize=11)
fig.savefig(args.out/f'observations-{args.year}.png',dpi=150)
plt.close(fig)
with (args.out/'observations-monthly.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['month','observed_systems'])
    w.writerows((m,monthly[m]) for m in range(1,13))
value=dict(source_url='https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_bytes=source.stat().st_size,year=args.year,basin='NA',track_type='main',status=['TS','HU','SS'],wind_field='USA_WIND',wind_units='kt, NHC 1-minute',minimum_wind_kt=args.minimum_wind_kt,cadence_hours=6,systems=len(tracks),points=len(points),monthly_counts=dict(monthly),peak_wind_kt=max(p['wind_kt'] for p in points),seconds=time.perf_counter()-start,model_available=False)
(args.out/'observations-receipt.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value,indent=2))
