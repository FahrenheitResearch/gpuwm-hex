#!/usr/bin/env python3
"""Prepare a fresh owned full-model oracle tree from pinned MPAS sources.

No reference source or binary is rebuilt in place. Compilation is separately
invoked as make -j4 (through the build queue on a shared host) with f0's recorded compiler/dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import difflib
from generate import inventory,overlay
from boundaries import generate_boundaries,generate_output_diagnostics
from replay.generate_replay import write_replays


def generator_hashes():
    here=Path(__file__).parent
    return {str(p.relative_to(here)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(here.rglob('*')) if p.suffix in {'.py','.F','.rs','.json'}}


def verify_generator(snapshot):
    # A live edit during v20 preparation could bind newer on-disk generator
    # bytes to already imported code. Refuse that mixed provenance.
    if generator_hashes()!=snapshot:
        raise ValueError('oracle generator changed during preparation; regenerate from one immutable revision')


def assert_build_idle(out):
    # Refreshing while a compiler reads this tree can produce a binary whose
    # inventory describes different sources. Check owned-tree builds only.
    root=out.resolve()
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            name=(process/'comm').read_text().strip()
            cwd=(process/'cwd').resolve(strict=True)
        except (OSError,RuntimeError):continue
        if name in {'make','gmake','gfortran','f951','ifx','fortcom','cc1','cc1plus'} and (cwd==root or root in cwd.parents):
            raise ValueError(f'owned build is active (PID {process.name}); refresh would race source/receipt generation')


def bind_reference_build_metadata(out, reference):
    """Use the audited native Registry version, never the oracle copy's Git status.

    The generated core setup differs when git describe sees our instrumentation
    edits. Preserve the reference's actual preprocessing input. This changes
    only an owned Makefile; it does not exclude any output or rebuild native.
    """
    audit_path=reference/'flag-audit.json'
    audit=json.loads(audit_path.read_text())
    if not audit['all_ok'] or audit['recipe']!='gnu-strict':
        raise ValueError('native build metadata requires the strict reference audit')
    versions=set()
    for row in audit['objects']:
        for command in row['commands']:
            for argument in command['argv']:
                if argument.startswith('-DMPAS_GIT_VERSION='):
                    versions.add(argument.split('=',1)[1])
    if len(versions)!=1:
        raise ValueError('native audit must bind one Registry git version')
    version=versions.pop()
    if not re.fullmatch(r'[A-Za-z0-9_.+-]+',version):
        raise ValueError('unsafe native Registry git version')
    makefile=out/'Makefile'
    text=makefile.read_text()
    pattern=r'^VER=\$\(shell git describe --dirty 2> /dev/null\)\n.*?^endif # END OF GIT DESCRIBE VERSION$'
    replacement=('# Oracle Registry metadata authenticated against the strict native audit.\n'
                 'override CPPFLAGS += -DMPAS_GIT_VERSION='+version)
    text,count=re.subn(pattern,lambda _:replacement,text,flags=re.M|re.S)
    if count!=1:
        raise ValueError('native Git version Makefile block is missing or ambiguous')
    makefile.write_text(text)
    return dict(git_version=version,audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())


def prepare(source,out,reference,refresh=False,pin_set_name="mpas-v842"):
    generator_snapshot=generator_hashes()
    if out.exists() and not refresh:raise ValueError('refusing existing build directory')
    if refresh:assert_build_idle(out)
    from pin_sets import reference_build,pin_set
    ref,upstream=reference_build(reference,pin_set_name)
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual!=upstream:raise ValueError('source/reference upstream pin differs')
    previous_files={}
    if refresh:
        previous=json.loads((out/'oracle-inventory.json').read_text())
        if previous['strict_reference_binary_sha256']!=ref['binary_sha256']:raise ValueError('owned build reference differs')
        paths=set(previous['overlay_paths'])|{b['path'] for b in previous['boundaries']}|{s['path'] for s in previous['module_state']}
        # A failed pass may have changed newly reached transitive modules before
        # publishing its inventory. Restore every candidate physics source, not
        # just paths remembered by the last successful pass.
        paths.update(str(p.relative_to(source)) for p in
                     (source/'src/core_atmosphere/physics').rglob('*')
                     if p.is_file() and p.suffix.lower() in {'.f','.f90'})
        # Restore these newly supported boundary modules even if a failed
        # generation did not finish writing its updated inventory.
        paths.update(['src/core_atmosphere/dynamics/mpas_atm_iau.F',
                      'src/core_atmosphere/mpas_atm_core.F'])
        paths.update(str(p.relative_to(source)) for p in
                     (source/'src/framework').glob('*')
                     if p.is_file() and p.suffix.lower() in {'.f','.f90'})
        paths.update(str(p.relative_to(source)) for p in
                     (source/'src/external/esmf_time_f90').glob('*.F90'))
        paths.update(['src/core_atmosphere/physics/mpas_atmphys_init.F','src/core_atmosphere/physics/Makefile','src/Makefile','Makefile'])
        paths.update(['src/core_atmosphere/physics/'+n for n in ('mpas_oracle_dump.F','mpas_oracle_pool.F','mpas_oracle_pool_restore.F','mpas_oracle_clock.F','mpas_oracle_timer.F','mpas_oracle_stream.F')])
        paths.update(previous.get('reference_plumbing_paths',[]))
        paths.update(previous.get('owned_build_paths',[]))
        previous_files={p:(hashlib.sha256((out/p).read_bytes()).hexdigest(),(out/p).stat())
                        for p in paths if (out/p).is_file()}
        for path in paths:
            if not (source/path).exists():continue
            target=out/path
            if target.is_symlink():raise ValueError(f'owned refresh target is a symlink: {target}')
            shutil.copy2(source/path,target)
    else:
        shutil.copytree(source,out,symlinks=True)
    # Pinned tree permissions are read-only; only this owned copy is changed.
    for p in [out,*out.rglob('*')]:
        if not p.is_symlink():p.chmod(p.stat().st_mode|0o200)
    plumbing=pin_set(pin_set_name).get('reference_plumbing')
    plumbing_paths=[]
    if plumbing:
        patch_text=(reference/plumbing['file']).read_text()
        for line in patch_text.splitlines():
            if line.startswith('+++ '):
                path=line[4:].split('\t',1)[0]
                if Path(path).is_absolute() or '..' in Path(path).parts or not (out/path).is_file():
                    raise ValueError('native reference plumbing escapes owned source files')
                plumbing_paths.append(path)
        subprocess.run(['patch','--batch','--forward','-p0','--directory',str(out)],input=patch_text,text=True,check=True,capture_output=True)
    manifest=inventory(source,pin_set_name)
    if plumbing:
        manifest['reference_build_metadata']=bind_reference_build_metadata(out,reference)
    overlay(source,out,manifest)
    # overlay only generated driver slices; boundary pass reads untouched paths.
    for rel in ['src/core_atmosphere/physics/mpas_atmphys_driver.F','src/core_atmosphere/dynamics/mpas_atm_time_integration.F']:
        (out/rel).unlink()
    generate_boundaries(source,out,manifest)
    (out/'src/core_atmosphere/mpas_atm_core.F').unlink()
    generate_output_diagnostics(source,out,manifest)
    from state import generate_state
    generate_state(source,out,manifest)
    from local_state import generate_locals
    generate_locals(source,out,manifest)
    from allocation_state import generate_allocations,generate_automatic_entries,generate_entry_storage
    if pin_set_name=='rrfs':generate_automatic_entries(source,out,manifest)
    generate_allocations(source,out,manifest)
    generate_entry_storage(source,out,manifest)
    write_replays(manifest,out/'oracle_replay')
    here=Path(__file__).parent
    physics=out/'src/core_atmosphere/physics'
    for name in ['mpas_oracle_dump.F','mpas_oracle_pool.F','mpas_oracle_pool_restore.F']:
        shutil.copy2(here/name,physics/name)
    from clock_graph import generate as generate_clock
    clock_source,manifest['clock_graph_schema']=generate_clock(source)
    (physics/'mpas_oracle_clock.F').write_text(clock_source)
    from timer_graph import generate as generate_timer
    timer_source,manifest['timer_graph_schema']=generate_timer(source)
    (physics/'mpas_oracle_timer.F').write_text(timer_source)
    from stream_graph import generate as generate_stream
    stream_source,manifest['stream_graph_schema']=generate_stream(source)
    (physics/'mpas_oracle_stream.F').write_text(stream_source)
    # Use the published reference's runtime tables. The upstream make target
    # otherwise fetches a fresh MPAS-Data checkout while compiling this copy.
    table_source=reference/'src/core_atmosphere/physics/physics_wrf/files'
    table_target=physics/'physics_wrf/files'
    table_target.mkdir(parents=True,exist_ok=True)
    manifest['reference_table_sha256']={}
    for p in sorted(table_source.iterdir()):
        if not p.is_file():continue
        target=table_target/p.name
        if target.is_symlink():target.unlink()
        shutil.copy2(p,target)
        manifest['reference_table_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    owned_build_paths=configure_owned_build(out,manifest)
    manifest['owned_build_paths']=owned_build_paths
    manifest['reference_plumbing_paths']=plumbing_paths
    manifest['strict_reference_manifest_sha256']=hashlib.sha256((reference/'MANIFEST.json').read_bytes()).hexdigest()
    manifest['strict_reference_binary_sha256']=ref['binary_sha256']
    manifest['reference_toolchain']=(reference/'toolchain.txt').read_text()
    paths=set(manifest['overlay_paths'])|{b['path'] for b in manifest['boundaries']}|{m['path'] for m in manifest['module_state']}
    paths.update(['src/Makefile','src/core_atmosphere/physics/mpas_atmphys_init.F','src/core_atmosphere/Registry.xml',
                  'src/core_atmosphere/physics/physics_noahmp/drivers/mpas/NoahmpIOVarType.F90',
                  'src/core_atmosphere/physics/physics_noahmp/drivers/mpas/NoahmpIOVarInitMod.F90',
                  'src/framework/mpas_timekeeping_types.inc'])
    paths.update(str(p.relative_to(source)) for p in (source/'src/external/esmf_time_f90').glob('*.F90'))
    paths.update(owned_build_paths)
    paths.update(manifest['timer_graph_schema']['source_sha256'])
    paths.update(manifest['stream_graph_schema']['source_sha256'])
    manifest['source_sha256'].update({p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in sorted(paths)})
    manifest['generated_sha256']={p:hashlib.sha256((out/p).read_bytes()).hexdigest() for p in sorted(paths) if (out/p).is_file()}
    manifest['generated_sha256']['src/core_atmosphere/physics/mpas_oracle_clock.F']=hashlib.sha256(clock_source.encode()).hexdigest()
    manifest['generated_sha256']['src/core_atmosphere/physics/mpas_oracle_timer.F']=hashlib.sha256(timer_source.encode()).hexdigest()
    manifest['generated_sha256']['src/core_atmosphere/physics/mpas_oracle_stream.F']=hashlib.sha256(stream_source.encode()).hexdigest()
    verify_generator(generator_snapshot)
    manifest['generator_sha256']=generator_snapshot
    # MPAS wrappers can be generated/ignored files in pinned externals. A Git
    # diff omits those edits; compare every generated source against its pin.
    patch=''
    paths.update(plumbing_paths+['Makefile','src/core_atmosphere/physics/Makefile'])
    paths.update('src/core_atmosphere/physics/'+n for n in ('mpas_oracle_dump.F','mpas_oracle_pool.F','mpas_oracle_pool_restore.F','mpas_oracle_clock.F','mpas_oracle_timer.F','mpas_oracle_stream.F'))
    for relative in sorted(paths):
        original=source/relative;generated=out/relative
        if not generated.is_file():continue
        old=original.read_text().splitlines(True) if original.exists() else []
        patch+=''.join(difflib.unified_diff(old,generated.read_text().splitlines(True),fromfile='a/'+relative if old else '/dev/null',tofile='b/'+relative))
    manifest['instrumentation_sha256']=hashlib.sha256(patch.encode()).hexdigest()
    (out/'oracle-instrumentation.diff').write_text(patch)
    (out/'oracle-inventory.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for relative,(digest,stat) in previous_files.items():
        p=out/relative
        if hashlib.sha256(p.read_bytes()).hexdigest()==digest:
            import os
            os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    return manifest


def configure_owned_build(out,manifest=None):
    physics=out/'src/core_atmosphere/physics'
    makefile=physics/'Makefile'
    text=makefile.read_text()
    text=text.replace('\t./../tools/manage_externals/checkout_externals --externals ./../Externals.cfg\n',
                      '\t@echo "Oracle sources and pinned externals were verified offline"\n')
    text=text.replace('\t./checkout_data_files.sh\n',
                      '\t@test -f physics_wrf/files/COMPATIBILITY || { echo "Pinned native tables must be staged before the full build"; exit 1; }\n')
    text=text.replace('core_physics_noahmp:', 'core_physics_noahmp: core_physics_init',1)
    text=text.replace('OBJS_init = \\\n','OBJS_init = \\\n\tmpas_oracle_dump.o \\\n\tmpas_oracle_clock.o \\\n\tmpas_oracle_timer.o \\\n\tmpas_oracle_stream.o \\\n\tmpas_oracle_pool.o \\\n\tmpas_oracle_pool_restore.o \\\n',1)
    text+='\n# Generated oracle module ordering; no numerical flags changed.\n'
    # Package submakes all append to this same archive. Their native targets
    # lack a common archive dependency, so concurrent ar invocations truncate
    # members. Serialize this make instance; each package still uses -j4.
    text+='\n.NOTPARALLEL:\n'
    text+='core_microphysics core_mynnedmf core_mynnsfc core_RUCLSM core_GFL core_SMOKE core_UGWP: core_physics_init\n'
    text+='mpas_oracle_pool.o mpas_oracle_pool_restore.o mpas_oracle_clock.o mpas_oracle_timer.o mpas_oracle_stream.o: mpas_oracle_dump.o\n'
    text+='$(filter-out mpas_oracle_dump.o mpas_oracle_clock.o mpas_oracle_timer.o mpas_oracle_stream.o mpas_oracle_pool.o mpas_oracle_pool_restore.o,$(OBJS_init)) $(OBJS): mpas_oracle_dump.o mpas_oracle_clock.o mpas_oracle_timer.o mpas_oracle_stream.o mpas_oracle_pool.o mpas_oracle_pool_restore.o\n'
    # The initialization snapshot imports additional native module helpers.
    # Their objects must precede physics_init, regardless of the original
    # package object-list order. Package modules already precede phys_interface.
    if manifest is not None:
        wanted=set(manifest.get('init_state_modules',[]))
        directory=Path('src/core_atmosphere/physics')
        dependencies=sorted({Path(m['path']).stem+'.o' for m in manifest['module_state']
                             if m['name'] in wanted and Path(m['path']).parent==directory
                             and Path(m['path']).stem!='mpas_atmphys_init'})
        text+='mpas_atmphys_init.o: '+' '.join(dependencies)+'\n'
    makefile.write_text(text)
    linkfile=out/'src/Makefile'
    if linkfile.is_file():
        native_archives='-ldycore -lops -lframework'
        linktext=linkfile.read_text()
        if linktext.count(native_archives)!=1:
            raise ValueError('native final archive link changed')
        grouped='-Wl,--start-group '+native_archives+' -Wl,--end-group'
        if grouped not in linktext:
            linkfile.write_text(linktext.replace(native_archives,grouped,1))
    p=out/'Makefile';text=p.read_text()
    # Changed driver/dynamics boundaries snapshot transitive scheme state.
    # Those imports need the module directories even when the native caller
    # normally imports only a public wrapper module.
    module_dirs={physics.resolve()}
    if manifest is not None:
        module_dirs.update((out/Path(m['path']).parent).resolve()
                           for m in manifest['module_state'])
        # NOAA package builds copy MPAS/src wrappers to their package root,
        # where gfortran writes .mod files. Include those native build roots
        # as well as the original source directories.
        for directory in list(module_dirs):
            module_dirs.update(p for p in directory.parents
                               if p==physics.resolve() or physics.resolve() in p.parents)
    includes=' '.join('-I'+str(p) for p in sorted(module_dirs))
    text=text.replace('FCINCLUDES =\n',f'FCINCLUDES = {includes}\n',1)
    # Native packages may use GNU's implicit .F rule (RUCLSM's surface
    # diagnostics does). That rule consumes FFLAGS, without FCINCLUDES.
    # Forward the oracle module search path to those compilations too.
    text=text.replace('FFLAGS="$(FFLAGS)"', 'FFLAGS="$(FFLAGS) $(FCINCLUDES)"')
    head,tail=text.split('\ngfortran:',1)
    if '"FFLAGS_OPT = -O3 -ffp-contract=off' not in tail.split('\ngfortran-clang:',1)[0]:
        tail=tail.replace('"FFLAGS_OPT = -O3 ', '"FFLAGS_OPT = -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations ',1)
    p.write_text(head+'\ngfortran:'+tail)
    smiol=out/'src/external/SMIOL/Makefile'
    text=smiol.read_text()
    text=text.replace('$(FC) $(CPPINCLUDES) $(FFLAGS) -c $<','$(FC) $(CPPFLAGS) $(CPPINCLUDES) $(FFLAGS) -c $<')
    smiol.write_text(text)
    # A refreshed RUCLSM module makes GNU's built-in Modula-2 .mod.o rule
    # compete with the Fortran rule. Use only the package's Fortran suffixes,
    # including its .F surface diagnostics, in this owned build.
    ruc=physics/'physics_noaa/RUCLSM/MPAS/Makefile'
    if ruc.exists():
        text=ruc.read_text()
        text=text.replace('.SUFFIXES: .F90 .o', '.SUFFIXES:\n.SUFFIXES: .F90 .F .o',1)
        if '\n.F.o:' not in text:
            rule=text.split('\n.F90.o:',1)[1]
            text+='\n.F.o:'+rule.replace('$*.F90','$*.F')
        ruc.write_text(text)
    # NOAA and other packages inherit GNU's built-in .mod.o rule too. Cancel
    # that non-Fortran rule without removing native .F/.c compilation rules.
    # Wrapper-root Makefiles are copied from MPAS/ during the build; record
    # and patch their actual MPAS/ source instead of the disposable copy.
    owned_paths=[]
    for p in sorted(physics.rglob('Makefile*')):
        if not p.is_file() or (p.parent/'MPAS'/p.name).is_file():
            continue
        text=p.read_text()
        # The fork's GFL deep module imports common, but the native package
        # Makefile omits this prerequisite. Preserve parallel compilation
        # while ensuring the actual GNU module is available first.
        if p == physics/'physics_noaa/GFL/MPAS/Makefile':
            dependency='module_cu_gfl_deep.o: module_cu_gfl_common.o'
            if dependency not in text:
                text+='\n'+dependency+'\n'
        if '\n%.o: %.mod\n' not in text:
            text+='\n# GNU .mod files here are Fortran modules, not Modula-2.\n.mod.o:\n%.o: %.mod\n'
        p.write_text(text)
        owned_paths.append(str(p.relative_to(out)))
    cfg=out/'src/core_atmosphere/Externals.cfg'
    cfg.write_text(re.sub(r'(?m)^repo_url = .*$', 'repo_url = .',cfg.read_text()))
    if linkfile.is_file():owned_paths.append('src/Makefile')
    return owned_paths


def prepare_sources(source,out,pin_set_name="mpas-v842",full_tree=False):
    """Prepare owned instrumented modules without claiming a native reference build."""
    generator_snapshot=generator_hashes()
    from pin_sets import source_files
    from state import generate_state
    from local_state import generate_locals
    if out.exists():raise ValueError('source-only output must be fresh')
    source=source.resolve();out=out.resolve()
    manifest=inventory(source,pin_set_name)
    if full_tree:
        shutil.copytree(source,out,ignore=shutil.ignore_patterns('.git','*.o','*.mod','*.a','__pycache__'))
        for p in [out,*out.rglob('*')]:
            if not p.is_symlink():p.chmod(p.stat().st_mode|0o200)
        for rel in ['src/core_atmosphere/physics/mpas_atmphys_driver.F',
                    'src/core_atmosphere/dynamics/mpas_atm_time_integration.F',
                    'src/core_atmosphere/mpas_atm_core.F']:
            (out/rel).unlink()
    else:out.mkdir(parents=True)
    overlay(source,out,manifest)
    generate_boundaries(source,out,manifest)
    generate_output_diagnostics(source,out,manifest)
    generate_state(source,out,manifest)
    generate_locals(source,out,manifest)
    from allocation_state import generate_allocations,generate_automatic_entries,generate_entry_storage
    if pin_set_name=='rrfs':generate_automatic_entries(source,out,manifest)
    generate_allocations(source,out,manifest)
    generate_entry_storage(source,out,manifest)
    write_replays(manifest,out/'oracle_replay')
    files=source_files(source,pin_set_name)
    files+=list((source/'src/framework').glob('*.inc'))
    for path in files:
        target=out/path.relative_to(source)
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    here=Path(__file__).parent
    for name in ('mpas_oracle_dump.F','mpas_oracle_pool.F','mpas_oracle_pool_restore.F'):
        shutil.copyfile(here/name,out/'src/core_atmosphere/physics'/name)
    from clock_graph import generate as generate_clock
    clock_source,manifest['clock_graph_schema']=generate_clock(source)
    (out/'src/core_atmosphere/physics/mpas_oracle_clock.F').write_text(clock_source)
    from timer_graph import generate as generate_timer
    timer_source,manifest['timer_graph_schema']=generate_timer(source)
    (out/'src/core_atmosphere/physics/mpas_oracle_timer.F').write_text(timer_source)
    from stream_graph import generate as generate_stream
    stream_source,manifest['stream_graph_schema']=generate_stream(source)
    (out/'src/core_atmosphere/physics/mpas_oracle_stream.F').write_text(stream_source)
    if full_tree:configure_owned_build(out,manifest)
    manifest['full_tree']=full_tree
    manifest['source_only']=True
    manifest['forecast_proven']=False
    # Source-only framework builds need the same immutable source check as
    # reference-backed builds. Include ignored NOAA wrappers and replay files.
    generated={str(p.relative_to(out)):p for p in out.rglob('*')
               if p.is_file() and p.suffix.lower() in {'.f','.f90','.inc'}}
    # MPAS copies wrapper Makefiles into external module roots while building.
    # Bind the actual edited build inputs, whose contents must stay immutable.
    for relative in ('Makefile','src/Makefile','src/core_atmosphere/physics/Makefile',
                     'src/external/SMIOL/Makefile'):
        p=out/relative
        if p.is_file():generated[relative]=p
    manifest['generated_sha256']={relative:hashlib.sha256(p.read_bytes()).hexdigest()
                                  for relative,p in sorted(generated.items())}
    patch=''
    for relative,p in sorted(generated.items()):
        original=source/relative
        old=original.read_text().splitlines(True) if original.exists() else []
        patch+=''.join(difflib.unified_diff(old,p.read_text().splitlines(True),
                    fromfile='a/'+relative if old else '/dev/null',tofile='b/'+relative))
    manifest['instrumentation_sha256']=hashlib.sha256(patch.encode()).hexdigest()
    (out/'oracle-instrumentation.diff').write_text(patch)
    verify_generator(generator_snapshot)
    manifest['generator_sha256']=generator_snapshot
    (out/'oracle-inventory.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--sources-only',action='store_true',help='prepare instrumented modules for standalone work; no forecast proof')
    p.add_argument('--full-tree',action='store_true',help='include native build machinery for framework compilation without a published referee')
    p.add_argument('--pin-set',choices=['mpas-v842','rrfs'],default='mpas-v842')
    p.add_argument('--refresh-owned',action='store_true',help='regenerate an existing marked owned oracle tree; no concurrent build may be running')
    a=p.parse_args()
    if a.source.resolve()==a.out.resolve() or a.source.resolve() in a.out.resolve().parents:p.error('output must be outside pinned source')
    if a.sources_only:
        if a.reference or a.refresh_owned:p.error('source-only preparation requires fresh output and no reference')
        m=prepare_sources(a.source,a.out,a.pin_set,a.full_tree)
    else:
        if not a.reference:p.error('--reference is required for a full build')
        m=prepare(a.source,a.out,a.reference,a.refresh_owned,a.pin_set)
    print(json.dumps({'sites':len(m['sites']),'boundaries':len(m['boundaries']),'argument_complete':m['argument_complete']}))


if __name__=='__main__':main()
