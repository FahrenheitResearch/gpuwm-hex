"""Box CPU score charts and compact tables from this run's receipts."""
import os
from pathlib import Path
import csv,hashlib,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def main(phase):
    out=W/'products'/phase
    if not (out/'complete.json').is_file(): raise ValueError('products must close before summarizing')
    receipt=json.loads((W/'runs'/phase/'ensemble-receipt.json').read_text())
    members=json.loads((out/'member-scores.json').read_text())
    ensemble=json.loads((out/'ensemble-scores.json').read_text())
    runtime=[]
    for job in receipt['jobs']:
        runtime.append(dict(member=job['id'],returncode=job['returncode'],
            card_ordinal=job['card_ordinal'],visible_device=job['segments'][0]['visible_device'],
            door_minutes=job['wall_seconds']/60,integration_minutes=job.get('integration_seconds',0)/60,
            peak_whole_card_mib=job['peak_whole_card_mib'],
            resident_devices=json.dumps(job.get('resident_devices')),
            physics_flags=json.dumps(job.get('forecast_flags',[]))))
    write_csv(out/'member-runtime.csv',runtime)
    flat=[]; reliability=[]
    for row in ensemble:
        flat.append(dict(lead_hour=row['lead_hour'],members=row['members'],brier_35=row['brier_score_35'],
            mean_csi_35=row['mean']['csi_35'],maximum_csi_35=row['maximum']['csi_35'],
            mean_fss_20km=row['mean']['fss']['20'],mean_fss_50km=row['mean']['fss']['50'],
            max_fss_20km=row['maximum']['fss']['20'],max_fss_50km=row['maximum']['fss']['50'],
            mean_position_error_km=row['mean']['position_error_km'],
            maximum_position_error_km=row['maximum']['position_error_km'],
            mean_max_dbz_bias=row['mean']['max_reflectivity_bias_dbz'],
            maximum_max_dbz_bias=row['maximum']['max_reflectivity_bias_dbz'],
            weighted_mean_spread_dbz=row['weighted_mean_spread_dbz'],rms_spread_dbz=row['rms_spread_dbz']))
        reliability.extend(dict(lead_hour=row['lead_hour'],**bin) for bin in row['reliability'])
    write_csv(out/'ensemble-scores.csv',flat); write_csv(out/'reliability.csv',reliability)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for name in sorted({r['member'] for r in members}):
        rows=sorted([r for r in members if r['member']==name],key=lambda r:r['lead_hour'])
        hours=[r['lead_hour'] for r in rows]
        axes[0,0].plot(hours,[r['csi_35'] for r in rows],label=name,marker='.')
        axes[0,1].plot(hours,[r['fss']['50'] for r in rows],label=name,marker='.')
    hours=[r['lead_hour'] for r in ensemble]
    axes[1,0].plot(hours,[r['brier_score_35'] for r in ensemble],marker='o')
    axes[1,1].plot(hours,[r['weighted_mean_spread_dbz'] for r in ensemble],marker='o')
    for ax,title in zip(axes.flat,('CSI at 35 dBZ','FSS at 35 dBZ, 50 km square','Brier score at 35 dBZ','Mean sample spread, dBZ')):
        ax.set_title(title); ax.set_xlabel('Forecast lead hour'); ax.grid(alpha=.25)
    axes[0,0].legend(ncol=2,fontsize=8); fig.suptitle(f'{phase}: historical MRMS verification, cold-start GEFS')
    fig.savefig(out/'score-chart.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    names=[r['member'] for r in runtime]; x=list(range(len(names)))
    axes[0].bar([v-.2 for v in x],[r['door_minutes'] for r in runtime],width=.4,label='Door start to passed')
    axes[0].bar([v+.2 for v in x],[r['integration_minutes'] for r in runtime],width=.4,label='Integration')
    axes[0].set_ylabel('Minutes'); axes[0].legend(fontsize=8)
    axes[1].bar(x,[r['peak_whole_card_mib'] for r in runtime]); axes[1].set_ylabel('Peak whole-card MiB')
    for ax in axes: ax.set_xticks(x,names,rotation=45); ax.grid(axis='y',alpha=.25)
    fig.suptitle(f'{phase}: facts from this run, one member per card')
    fig.savefig(out/'runtime-chart.png',dpi=160); plt.close(fig)
    maps=list((out/'maps').rglob('*.png'))+list((out/'aggregate').rglob('*.png'))
    captions=[]; artifacts=[]
    for path in sorted(maps):
        with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        relative=path.relative_to(out)
        caption=f'{relative}: Rust weather product from the static 48 to 3 km cap; active one-card LTS; historical GEFS cold start. Product and valid time are printed on the map.'
        captions.append(caption)
        artifacts.append(dict(path=str(path),relative=str(relative),bytes=path.stat().st_size,sha256=sha))
    (out/'CAPTIONS.md').write_text('\n\n'.join(captions)+'\n')
    (out/'map-artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')

if __name__=='__main__': main(sys.argv[1])
