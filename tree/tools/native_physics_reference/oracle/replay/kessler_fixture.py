#!/usr/bin/env python3
"""Emit Fortran synthetic input construction and full pinned Kessler module.

MPAS src/core_atmosphere/physics/physics_wrf/module_mp_kessler.F:8-243.
Python emits source only; all field values and stream writes execute in Fortran.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from generate import dump_call
from generate_replay import generate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    m=json.loads(a.inventory.read_text())
    site=next(s for s in m['sites'] if s['scheme']=='kessler')
    # Kessler has no mutable module state; use the complete pristine module.
    site=dict(site,state_modules=[])
    replay=generate(site)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'replay.F90').write_text(replay)
    (a.out/'module_mp_kessler.F').write_bytes((a.source/site['callee']['path']).read_bytes())
    begin=replay.index('  if(command_argument_count()')
    invoke=replay.index('  call kessler(')
    finish=replay.index('  call oracle_clear_output()')
    decls=replay[:begin].replace('program replay_'+site['site'],'program capture_kessler')
    dims={'ims':1,'ime':4,'ids':1,'ide':5,'its':1,'ite':4,
          'jms':1,'jme':1,'jds':1,'jde':1,'jts':1,'jte':1,
          'kms':1,'kme':12,'kds':1,'kde':13,'kts':1,'kte':12}
    init="  integer :: i,k\n  real(real32) :: spacing\n  character(len=128) :: argument\n  if(command_argument_count()<1.or.command_argument_count()>3) error stop 'usage: capture OUTPUT_PREFIX [DT_SECONDS] [LAYER_THICKNESS]'\n  call get_command_argument(1,output)\n  dt_in=30.0_real32;spacing=300.0_real32\n  if(command_argument_count()>=2) then\n    call get_command_argument(2,argument)\n    read(argument,*) dt_in\n  endif\n  if(command_argument_count()>=3) then\n    call get_command_argument(3,argument)\n    read(argument,*) spacing\n  endif\n"
    init+=''.join(f'  {n}={v}\n' for n,v in dims.items())
    for arg in site['arguments']:
        if arg['dims']:
            init+=f"  allocate({arg['name']}({','.join(arg['dims'])}))\n"
    init+='''  xlv=2.5e6_real32; cp=1004.5_real32; ep2=0.622_real32
  svp1=0.6112_real32; svp2=17.67_real32; svp3=29.65_real32; svpt0=273.15_real32
  rhowater=1000.0_real32
  rainnc=0.0_real32; rainncv=0.0_real32
  do k=1,12
   do i=1,4
    t(i,k,1)=285.0_real32+real(k,real32)
    pii(i,k,1)=1.0_real32-real(k,real32)*0.01_real32
    rho(i,k,1)=1.1_real32-real(k,real32)*0.025_real32
    z(i,k,1)=real(k-1,real32)*spacing
    dz8w(i,k,1)=spacing
    qv(i,k,1)=0.003_real32*real(i,real32)
    qc(i,k,1)=0.0005_real32*real(i-1,real32)
    qr(i,k,1)=0.0001_real32*real(i-1,real32)
   enddo
  enddo
  call oracle_set_context([101_int32,205_int32,309_int32,413_int32],1,0)
'''
    init+=f"  call oracle_begin('{site['site']}',event)\n  call oracle_set_output(trim(output))\n"
    for arg in site['arguments']:
        if arg['intent']!='out':
            init+=dump_call(site,dict(arg,actual=arg['name']),'pre').replace('oracle_event','event')
    # The source's complete module is called unchanged; same output writes as replay.
    (a.out/'capture.F90').write_text(decls+init+replay[invoke:finish]+"  call oracle_clear_output()\nend program\n")
    (a.out/'Makefile').write_text('''FC = gfortran
RECIPE ?= gnu-strict
BASE = -std=f2008 -fimplicit-none -ffree-form -ffree-line-length-none -fconvert=big-endian
STRICT = -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations
ifeq ($(RECIPE),gnu-strict)
FLAGS = $(BASE) $(STRICT)
else ifeq ($(RECIPE),gnu-o3)
FLAGS = $(BASE) -O3
else ifeq ($(RECIPE),coverage)
FLAGS = $(BASE) $(STRICT) --coverage
else ifeq ($(RECIPE),ifx-o3)
FC = /opt/intel/oneapi/compiler/2026.0/bin/ifx
FLAGS = -free -extend-source -convert big_endian -O3 -align array64byte -Qoption,fpp,-macro_expand=vc
else ifeq ($(RECIPE),ifx-valuesafe)
FC = /opt/intel/oneapi/compiler/2026.0/bin/ifx
FLAGS = -free -extend-source -convert big_endian -O3 -align array64byte -Qoption,fpp,-macro_expand=vc -fp-model=source -no-fma -fimf-arch-consistency=true
else
$(error unsupported fixture recipe)
endif
all: capture replay oracle_words
mpas_oracle_dump.o: transport.F
	$(FC) $(FLAGS) -c $< -o $@
module_mp_kessler.o: module_mp_kessler.F
	$(FC) $(FLAGS) -c $< -o $@
capture: capture.F90 mpas_oracle_dump.o module_mp_kessler.o
	$(FC) $(FLAGS) $^ -o $@
replay: replay.F90 mpas_oracle_dump.o module_mp_kessler.o
	$(FC) $(FLAGS) $^ -o $@
oracle_words: oracle_words.rs
	rustc --edition=2021 -O $< -o $@
''')
    transport=Path(__file__).parents[1]
    (a.out/'transport.F').write_bytes((transport/'mpas_oracle_dump.F').read_bytes())
    (a.out/'oracle_words.rs').write_bytes((transport/'oracle_words.rs').read_bytes())
    print(site['site'])


if __name__=='__main__':main()
