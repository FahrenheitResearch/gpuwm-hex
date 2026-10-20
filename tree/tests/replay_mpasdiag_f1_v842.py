"""Whole diagnostic modules on sealed real f1 pool inputs, MPI1.

Rust decodes typed ORCL0001 records. Python only binds decoded buffers to the
native/device ABI. Cloud cases pair the real cloudiness POST with the same-step
physics_get_tend PRE geometry/pressure; PV/budget/sounding inputs are from that
one PRE snapshot. These are stated real module cases, not forecast-call replay.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import cupy as cp
import numpy as np
from hexcore.mpasdiag.cloud_v842 import CloudDiagnostics, FIELDS
from hexcore.mpasdiag.pv_v842 import PVDiagnostics
from hexcore.mpasdiag.soundings_v842 import Soundings
from test_mpasdiag_v842 import native_library, F

class Metadata(ct.Structure):
    _fields_ = [('header', ct.c_int32*21), ('strings', ct.c_uint8*768),
               ('payload', ct.POINTER(ct.c_uint8)), ('payload_bytes', ct.c_size_t),
               ('ids', ct.POINTER(ct.c_uint8))]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('bank', type=Path); p.add_argument('native', type=Path)
    p.add_argument('decoder', type=Path); p.add_argument('receipt', type=Path)
    a = p.parse_args()
    manifest = json.loads((a.bank/'index.json').read_text())
    decoder = ct.CDLL(str(a.decoder.resolve()))
    decoder.oracle_open.argtypes=[ct.c_char_p];decoder.oracle_open.restype=ct.c_void_p
    decoder.oracle_metadata.argtypes=[ct.c_void_p,ct.POINTER(Metadata)]
    decoder.oracle_close.argtypes=[ct.c_void_p]
    native = native_library(a.native.resolve())
    result = dict(schema='o4-real-native-tap-module-replay-v1', passed=False,
                  process_count=1, forecast_proof=False, exclusions=[], cases=[],
                  case_scope=manifest['case_scope'],
                  native_scratch='Measured strict-binary allocator words transported into standalone ABI; numerical expressions unchanged.',
                  input_index_sha256=hashlib.sha256((a.bank/'index.json').read_bytes()).hexdigest(),
                  native_sha256=hashlib.sha256(a.native.read_bytes()).hexdigest(),
                  decoder_sha256=hashlib.sha256(a.decoder.read_bytes()).hexdigest())
    def compare(rows, name, expected, actual):
        want=cp.asarray(expected,order='F');got=actual
        assert want.shape==got.shape,(name,want.shape,got.shape)
        different=int(cp.count_nonzero(want.view(cp.uint32)!=got.view(cp.uint32)).get())
        rows.append(dict(field=name, words=want.size, differing_words=different))
        a.receipt.write_text(json.dumps(result,indent=2)+'\n')
        assert different==0,rows[-1]
    for case in manifest['cases']:
        handles=[]; values={}
        record=dict(suite=case['suite'],step=case['step'],source_root_sha256=case['source_root_sha256'],
                    input_members=len(case['records']),rows=[])
        result['cases'].append(record); rows=record['rows']
        try:
            for key, info in case['records'].items():
                path=a.bank/info['file']
                assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
                handle=decoder.oracle_open(str(path.resolve()).encode());assert handle,key
                handles.append(handle);meta=Metadata()
                assert decoder.oracle_metadata(handle,ct.byref(meta))==0
                h=list(meta.header); assert h[0] in (1,3), (key,h)
                assert h[13]==case['step'] and h[15]==0, (key,h[13:16])
                shape=tuple(h[3:3+h[2]])
                ctype=ct.c_float if h[0]==1 else ct.c_int32
                view=np.ctypeslib.as_array(ct.cast(meta.payload,ct.POINTER(ctype)),
                                         shape=(meta.payload_bytes//4,)).reshape(shape,order='F')
                values[key]=view
            scalar=lambda key:int(values[key].item())
            nc,nz,ne,nv,me,ns=[scalar('mesh.'+n) for n in
                    ('nCells','nVertLevels','nEdges','nVertices','maxEdges','nCellsSolve')]
            assert nc==ns, 'This replay fixes a complete MPI1 owned domain.'
            record.update(nCells=nc,nCellsSolve=ns,nVertLevels=nz,nEdges=ne,nVertices=nv)
            geom=('nEdgesOnCell','cellsOnCell','edgesOnCell','cellsOnEdge','verticesOnCell',
                  'cellsOnVertex','kiteAreasOnVertex','areaCell','dvEdge','edgeNormalVectors',
                  'cellTangentPlane','localVerticalUnitVectors','zgrid','latCell',
                  'verticesOnEdge','dcEdge','areaTriangle','coeffs_reconstruct','zz')
            h={n:values['mesh.'+n] for n in geom}
            m={n:cp.asarray(v,order='F') for n,v in h.items()}
            m.update(nCells=nc,nCellsSolve=ns,nVertLevels=nz,nEdges=ne,nVertices=nv,maxEdges=me)
            dev=lambda key:cp.asarray(values[key],order='F')
            qv=values['state.scalars.t1'][scalar('state.index_qv')-1,:,:]
            tm=values['state.theta_m.t1'];rz=values['state.rho_zz.t1'];w=values['state.w.t1']
            d={n:dev('diag.'+n) for n in ('theta','rho','ertel_pv','iLev_DT','inTropo',
               'uReconstructX','uReconstructY','uReconstructZ','uReconstructZonal','uReconstructMeridional',
               'vorticity','u_pv','v_pv','theta_pv','vort_pv','rho_edge','tend_u_phys',
               'dtheta_dt_mp','dtheta_dt_mix', 'depv_dt_diab_pv','depv_dt_fric_pv')}
            s=dict(w=dev('state.w.t1'),theta_m=dev('state.theta_m.t1'),
                   rho_zz=dev('state.rho_zz.t1'),qv=cp.asarray(qv,order='F'))
            port=PVDiagnostics(m,s,d)
            want_theta=F(values['diag.theta']);want_rho=F(values['diag.rho'])
            unused=F(np.zeros_like(want_theta))
            native('replay_prepare',nc*nz,F(tm),F(qv),F(rz),h['zz'],want_theta,want_rho,unused)
            port._kernel('prepare',nc*nz,(cp.int32(nc*nz),s['theta_m'],s['qv'],s['rho_zz'],m['zz'],
                         port.rvord,d['theta'],d['rho']))
            compare(rows,'theta',want_theta,d['theta']);compare(rows,'rho',want_rho,d['rho'])
            mesh_args=[h[n] for n in geom[:13]]
            epv=F(values['diag.ertel_pv'])
            native('replay_mesh',nc,nz,me,nv,ne,*mesh_args,ns,nz,
                   values['diag.uReconstructX'],values['diag.uReconstructY'],values['diag.uReconstructZ'],
                   w,values['diag.vorticity'],want_rho,want_theta,epv)
            port.calc_epv();compare(rows,'ertel_pv',epv,d['ertel_pv'])
            lev=F(values['diag.iLev_DT'],np.int32);mask=F(values['diag.inTropo'],np.int32)
            native('replay_tropo',nc,nz,me,h['nEdgesOnCell'],h['cellsOnCell'],h['latCell'],
                   epv,lev,mask,np.float32(2),np.float32(10))
            port.flood_fill_tropo()
            compare(rows,'iLev_DT',lev,d['iLev_DT']);compare(rows,'inTropo',mask,d['inTropo'])
            def interp(name, field, output):
                # mpas_pv_diagnostics.F:1154-1156: strict native prefix words remain zero.
                expected=F(values['diag.'+name])
                native('replay_interp_prefix',nc,nz,h['latCell'],lev,
                       F(np.r_[np.float32(0),epv.ravel(order='F')]),
                       F(np.r_[np.float32(0),field.ravel(order='F')]),np.float32(2),expected)
                port.interp_pv(cp.asarray(field,order='F'),output)
                compare(rows,name,expected,output)
            for name,field in [('u_pv',values['diag.uReconstructZonal']),
                               ('v_pv',values['diag.uReconstructMeridional']),
                               ('theta_pv',want_theta),('vort_pv',epv)]:interp(name,field,d[name])
            td={n:dev('tend.'+n) for n in ('u_euler','w_euler','theta_euler')}
            tp={n:dev('tend_physics.'+n) for n in ('rthratenlw','rthratensw','rthblten','rthcuten')}
            port.tend=td;port.tend_physics=tp
            mixed=F(values['diag.dtheta_dt_mix']);unused_a=F(np.zeros_like(mixed));unused_b=unused_a.copy(order='F')
            native('replay_prepare',nc*nz,F(tm),F(qv),F(values['tend.theta_euler']),h['zz'],unused_a,unused_b,mixed)
            port._kernel('mix_theta',nc*nz,(cp.int32(nc*nz),td['theta_euler'],s['qv'],port.rvord,d['dtheta_dt_mix']))
            compare(rows,'dtheta_dt_mix',mixed,d['dtheta_dt_mix'])
            schemes=('lw','sw','bl','cu','mp','mix','diab','fric')
            outputs=F(np.stack([values['diag.depv_dt_'+n] for n in schemes],axis=2))
            terms=F(np.stack([values['tend_physics.'+n] for n in
                     ('rthratenlw','rthratensw','rthblten','rthcuten')]+[values['diag.dtheta_dt_mp'],mixed],axis=2))
            for name in schemes:d['depv_dt_'+name]=dev('diag.depv_dt_'+name)
            physical=F(values['diag.tend_u_phys']);ew=F(values['tend.w_euler'])
            native('replay_full_budget',nc,nz,me,nv,ne,ns,*mesh_args,h['verticesOnEdge'],h['dcEdge'],
                   h['areaTriangle'],h['coeffs_reconstruct'],want_theta,want_rho,
                   values['diag.uReconstructX'],values['diag.uReconstructY'],values['diag.uReconstructZ'],w,
                   values['diag.vorticity'],F([1]*6,np.int32),terms,values['tend.u_euler'],
                   values['diag.rho_edge'],physical,ew,outputs)
            port.calc_pv_budget()
            for i,name in enumerate(schemes):compare(rows,'depv_dt_'+name,outputs[:,:,i],d['depv_dt_'+name])
            compare(rows,'tend_u_phys',physical,d['tend_u_phys']);compare(rows,'w_euler',ew,td['w_euler'])
            interp('depv_dt_diab_pv',outputs[:,:,6],d['depv_dt_diab_pv'])
            interp('depv_dt_fric_pv',outputs[:,:,7],d['depv_dt_fric_pv'])
            cloud={n:dev('diag.'+n) for n in ('pressure_p','pressure_base',*FIELDS)}
            wants=[F(values['diag.'+n]) for n in FIELDS]
            native('replay_cloud',ns,nz,values['diag.pressure_p'],values['diag.pressure_base'],
                   values['cloud.cldfrac'],*wants)
            CloudDiagnostics(m,cloud,{'cldfrac':dev('cloud.cldfrac')}).compute(FIELDS)
            for name,want in zip(FIELDS,wants):compare(rows,name,want,cloud[name])
            arrays=[tm,F(qv),values['diag.exner'],values['diag.pressure_base'],values['diag.pressure_p'],
                    values['diag.uReconstructZonal'],values['diag.uReconstructMeridional'],h['zgrid']]
            cells=F(np.arange(1,ns+1),np.int32);owned=F(np.ones(ns),np.int32)
            expected=F(np.zeros((6,nz,ns)));actual=cp.empty_like(cp.asarray(expected,order='F'))
            native('replay_sounding',ns,nz,nc+1,cells,owned,*arrays,expected)
            sounding=Soundings()
            sounding._kernel('sounding')(((ns*nz+127)//128,),(128,),
                (cp.int32(ns),cp.int32(nz),cp.asarray(cells),cp.asarray(owned),
                 *(cp.asarray(v,order='F') for v in arrays),sounding.rvord,sounding.pi,actual))
            compare(rows,'sounding-six-profile-values',expected,actual)
            record['passed']=True
            print(case['suite'],case['step'],'all output words equal',flush=True)
        finally:
            for handle in handles:decoder.oracle_close(handle)
    result['passed']=len(result['cases'])==len(manifest['cases']) and all(c.get('passed') for c in result['cases'])
    result['compared_words']=sum(r['words'] for c in result['cases'] for r in c['rows'])
    a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
