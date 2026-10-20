"""Plot computational class ceilings. This is a mesh-policy analysis map."""
from pathlib import Path
import numpy as np


def plot_rates(grid, rates, infeasible, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from netCDF4 import Dataset
    with Dataset(grid) as dataset:
        lat=np.rad2deg(np.asarray(dataset['latCell'][:]))
        lon=np.rad2deg(np.asarray(dataset['lonCell'][:]))
    lon=(lon+180)%360-180
    levels=np.rint(np.log2(rates)).astype(int)
    colors=ListedColormap(['#487aa0','#55a397','#a1bf60','#d8b463','#d16c59'])
    fig,ax=plt.subplots(figsize=(11,5.5))
    points=ax.scatter(lon,lat,c=levels,s=.5,cmap=colors,norm=BoundaryNorm(np.arange(-.5,5.5),5),rasterized=True,linewidths=0)
    if np.any(infeasible):
        ax.scatter(lon[infeasible],lat[infeasible],s=3,c='#3d1626',rasterized=True,linewidths=0,label='18 s base fails donor target')
        ax.legend(loc='lower left',fontsize=8)
    bar=fig.colorbar(points,ax=ax,ticks=np.arange(5),shrink=.85)
    bar.ax.set_yticklabels(['1','2','4','8','16'])
    bar.set_label('Optimistic transport class rate')
    ax.set(xlim=(-180,180),ylim=(-90,90),xlabel='Longitude, degrees',ylabel='Latitude, degrees',title='60 to 3 km: mesh-policy ceilings after the dry shadow pilot')
    ax.grid(alpha=.15)
    fig.text(.5,.035,'Original UCAR mesh, fine core at 0 N, 0 E. Classes use actual dry-stage donor fluxes and two buffer rings.\nAdvection, acoustics, mixing and physics can only lower these ceilings. No active LTS forecast was run.',ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.22,right=.93)
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=180)
    plt.close(fig)
