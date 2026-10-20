"""Resolve the fork's complete scientific object lists to pristine MPAS sources."""
import re

# Physics directories a licence stub replaces (build.py --licence-stub): neither fetched nor compiled, so
# their Makefiles are not read.  Empty for a lab build, which compiles everything.
HELD = set()


def objects(makefile):
    text = re.sub(r'\\\n', ' ', makefile.read_text())
    return [x for m in re.finditer(r'^OBJS(?:_init)?\s*=\s*(.*)$', text, re.M)
            for x in re.findall(r'[\w.]+\.o', m[1])]


def physics_sources(source):
    physics = source / 'src/core_atmosphere/physics'
    dirs = [physics, physics / 'physics_wrf', physics / 'physics_mmm',
            physics / 'physics_noaa/UGWP', physics / 'physics_noaa/TEMPO',
            *[physics / 'physics_noaa' / name for name in
              ['MYNN-EDMF', 'MYNN-SFC', 'RUCLSM', 'GFL', 'SMOKE']], physics / 'physics_noahmp/utility',
            physics / 'physics_noahmp/src', physics / 'physics_noahmp/drivers/mpas']
    dirs = [d for d in dirs if d == physics or d.relative_to(physics).parts[0] not in HELD]
    chosen = []
    for d in dirs:
        mf = d / ('Makefile.mpas' if d.name in {'physics_mmm', 'TEMPO'} else
                  'MPAS/Makefile' if d.name in {'MYNN-EDMF', 'MYNN-SFC', 'RUCLSM', 'GFL', 'SMOKE'} else 'Makefile')
        for obj in objects(mf):
            # Upstream sub-makes copy MPAS adapters into the scheme root. Select
            # their pristine originals without writing to the reference tree.
            paths = [directory / (obj[:-2] + suffix)
                     for directory in [d / 'MPAS', d / 'src', d]
                     for suffix in ['.F', '.F90', '.f90']]
            found = next((p for p in paths if p.exists()), None)
            if found is None:
                raise SystemExit(f'missing pristine source for {mf}: {obj}')
            chosen.append(found)
    return sorted(set(chosen))
