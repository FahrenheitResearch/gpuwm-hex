#!/usr/bin/env python3
"""Validate a completed identity_compare receipt before permitting case caching."""
import argparse
import json
from pathlib import Path

FIELDS = {'rho','rho_theta','rho_u','rho_w','scalars','theta_m','exner',
          'density_perturbation','rho_theta_perturbation','pressure_perturbation',
          'normal_velocity','vertical_velocity'}

def validate(receipt,hours,cells,levels,init_sha256):
    errors=[]
    if receipt.get('schema')!='gpuwm-hex.identity-extend/v1':errors.append('unexpected receipt schema')
    if receipt.get('partial') or receipt.get('native_finished') is not True:errors.append('native comparison incomplete')
    if receipt.get('all_identical') is not True:errors.append('identity failed')
    for key,value in (('cells',cells),('levels',levels),('init_sha256',init_sha256)):
        if receipt.get(key)!=value:errors.append(f'{key} mismatch')
    rows=receipt.get('rows',[])
    if [r.get('hours') for r in rows]!=hours:errors.append('required output times not compared')
    for row in rows:
        fields=row.get('fields',{})
        if set(fields)!=FIELDS:errors.append(f'field coverage incomplete at {row.get("hours")} h')
        if row.get('differing_words_total')!=0:errors.append(f'words differ at {row.get("hours")} h')
        if any(f.get('identical') is not True or f.get('words',0)<=0 for f in fields.values()):
            errors.append(f'field comparison failed at {row.get("hours")} h')
    if not hours:errors.append('empty required time set')
    return errors

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('receipt',type=Path);p.add_argument('--hours',type=float,nargs='+',required=True)
    p.add_argument('--cells',type=int,required=True);p.add_argument('--levels',type=int,required=True)
    p.add_argument('--init-sha256',required=True)
    a=p.parse_args();errors=validate(json.loads(a.receipt.read_text()),a.hours,a.cells,a.levels,a.init_sha256)
    print(json.dumps({'acceptance':not errors,'errors':errors}))
    return int(bool(errors))
if __name__=='__main__':raise SystemExit(main())
