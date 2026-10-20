#!/usr/bin/env python3
"""Allocate the five condensate species with physics suite 'none'.

Native MPAS-A v8.4.2 Registry.xml ties qc, qr, qi, qs and qg to microphysics,
PBL and convection packages, so a physics-off run carries qv alone. This edit
removes ONLY the packages attribute of those species, in every var_array that
carries the moist group: scalars, scalars_tend (tend_q*), lbc_scalars (lbc_q*)
and the ambient copy (q*_amb): 20 attributes. A physics-off run then
allocates, reads, advects (same monotonic routine) and mass-loads them exactly
as it does qv. No equation, routine or constant changes; number concentrations
and every other package are untouched.

All four arrays must be edited together. The solver indexes scalars_tend with
the scalars pool's index_q*; the first attempt (2026-10-11) edited scalars
alone, left scalars_tend one species long, and the binary read past it (qv max
6,329 kg/kg after one step). Refuses unless exactly 20 attributes are removed.
"""
import re, sys
path = sys.argv[1]
text = open(path, encoding='utf-8').read()
count = 0
def strip(match):
    global count
    stripped, n = re.subn(r'\s*packages="[^"]*"', '', match.group(0))
    count += n
    return stripped
pattern = re.compile(r'<var name="(?:tend_|lbc_)?(?:qc|qr|qi|qs|qg)(?:_amb)?"\s+'
                     r'(?:name_in_code="(?:qc|qr|qi|qs|qg)"\s+)?array_group="moist"[^>]*/>', re.S)
text = pattern.sub(strip, text)
if count != 20:
    raise SystemExit(f'expected 20 condensate packages attributes (5 species x 4 arrays), removed {count}')
open(path, 'w', encoding='utf-8').write(text)
print('condensate packages removed: 20')
