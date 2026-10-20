"""Box-only analysis chart of the design's measured launch proxy."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--projection60',type=Path,required=True)
    p.add_argument('--projection15',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    kernels=['acoustic_rs_ts','acoustic_ru','acoustic_column_solve']
    labels=['cell acoustic RHS','edge momentum','vertical solve']
    colors=['#356a95','#58a09a','#ca9055']
    for ax,path,title in zip(axes,[args.projection60,args.projection15],['60 to 3 km, one card','15 to 3 km, one card']):
        mesh=json.loads(path.read_text())['meshes'][0]
        ladder=mesh['ladders'][0]
        rates=ladder['rates_present']
        totals=[0.]*(len(rates)+1)
        for kernel,label,color in zip(kernels,labels,colors):
            costs=[mesh['full_mesh_ms'][kernel]]+[ladder['per_kernel'][kernel]['class_ms'][str(rate)]/rate for rate in rates]
            ax.bar(range(len(costs)),costs,bottom=totals,label=label,color=color,width=.7)
            totals=[a+b for a,b in zip(totals,costs)]
        ax.set_xticks(range(len(totals)),['fixed']+[f'rate {rate}' for rate in rates])
        ax.set_ylabel('Measured launch proxy, ms per base tick')
        ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
        for x,value in enumerate(totals):
            ax.text(x,value+max(totals)*.02,f'{value:.2f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(totals)*1.2)
        ax.text(.97,.96,f'Fixed / sum of classes: {totals[0]/sum(totals[1:]):.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=9)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.035,'Design probes, 2026-10-11. Each class cost is weighted by 1/rate.\nTransport, physics, interfaces and halos are absent. These are not forecast-hour timings.',ha='center',fontsize=10)
    fig.subplots_adjust(bottom=.23,top=.82,wspace=.28)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
