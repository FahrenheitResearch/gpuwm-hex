"""Pinned MPAS GFL driver ABI metadata; shared only by development tools."""
from pathlib import Path
import re

INT_NAMES = {'itimestep','ichoice_deep','ichoice_shallow','ishallow_g3','sub3d','num_chem',
             'kpbl','bilbc','k22_shallow','kbcon_shallow','ktop_shallow'}
SCALARS = {'itimestep','dt','xlv','cp','g','r_v','ichoice_deep','ichoice_shallow','ishallow_g3','sub3d','num_chem'}
SURFACES = {'dxcell','areacell','ht','xland','hfx','qfx','gsw','kpbl','bilbc','htop','hbot','k22_shallow',
            'kbcon_shallow','ktop_shallow','xmb_total','xmb_shallow','raincv','pratec','vcpool','sig_deep',
            'sig_deep_far','maxmf','rainncv'}
DIMENSIONS = ('ims','ime','jms','jme','kms','kme','ids','ide','jds','jde','kds','kde','its','ite','jts','jte','kts','kte')
OUTPUTS = ('htop','hbot','k22_shallow','kbcon_shallow','ktop_shallow','xmb_total','xmb_shallow',
           'raincv','pratec','gdc','gdc2','rthcuten','rqvcuten','rqccuten','rqicuten','rucuten','rvcuten',
           'qc3d','qi3d','chem3d_in','qc_cu','qi_cu','cldfrac_cu','vcpool','sig_deep','sig_deep_far',
           'sub3d_rthcuten','sub3d_rqvcuten','sub3d_rucuten','sub3d_rvcuten','refl10cm_cu')


def arguments(path):
    text=Path(path).read_text()
    match=re.search(r'subroutine cu_grell_freitas_li\((.*?)\)', text, re.I|re.S)
    return tuple(re.sub(r'[&\s]', '', match[1]).lower().split(','))
