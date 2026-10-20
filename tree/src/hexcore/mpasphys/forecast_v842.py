"""MPAS suite entry at the real forecast door; mpas_atm_core.F:589-596,1036-1040."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from datetime import datetime, timedelta
from fractions import Fraction
import json
import os
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from ..config import parse_fortran_namelist
from ..physics_suites import load_table
from importlib import import_module
from .driver_v842 import interval_seconds
from .pool_v842 import decode_registry_pools


def integral_duration(duration, dt):
    """Resolve CLI float hours to whole steps with the forecast door's tolerance.

    Five minutes arrives through argparse as 0.08333333333333333 hours.
    Its decimal Fraction is just below 300 seconds. Snap only within 1e-9
    of a step, as the regular forecast door does; reject real partial steps.
    """
    if duration is None or dt <= 0 or duration <= 0:
        raise ValueError('native run duration must be a positive integral number of config_dt steps')
    steps = duration / dt
    nearest = round(steps)
    if nearest <= 0 or abs(steps - nearest) > Fraction(1, 10**9):
        raise ValueError('native run duration must be an integral number of config_dt steps')
    return nearest * dt


@dataclass(frozen=True)
class Stream:
    name: str
    directions: tuple[str, ...]
    filename: str
    input_interval: str
    output_interval: str
    fields: tuple[str, ...]
    filename_interval: str = "none"
    input_file: str | None = None
    # Retain precision, clobber and file-format controls for the Rust writer.
    options: dict | None = None


def stream_input_bindings(values):
    """Bind native input files explicitly; mpas_atmphys_init.F:451-475."""
    result = {}
    for value in values or ():
        name, separator, filename = value.partition('=')
        if not separator or not name or not filename:
            raise ValueError('stream input requires NAME=FILE')
        if name in result:
            raise ValueError(f'duplicate native stream input {name}')
        path = Path(filename).resolve()
        if not path.is_file():
            raise ValueError(f'native stream input {name} does not exist: {path}')
        result[name] = path
    return result


def bind_stream_inputs(streams, bindings):
    """Keep native input and output paths distinct; mpas_atmphys_init.F:451-475."""
    known = {stream.name:stream for stream in streams}
    for name in bindings:
        if name not in known or 'input' not in known[name].directions:
            raise ValueError(f'native stream input {name} is not an active input stream')
    return tuple(replace(stream,input_file=str(bindings[stream.name]))
                 if stream.name in bindings else stream for stream in streams)


_MODE_KEYS = ('config_physics_mode', 'config_physics_correct_skip', 'config_numerics')


def split_physics_mode(text: str) -> tuple[str, str | None, tuple[str, ...]]:
    """Take the correct-mode keys out of &nhyd_model (MODE-KEY.md).

    The pristine MPAS v8.4.2 Registry has no such options, so the native pool decoder and the native
    library must never see them. A namelist without them is returned unchanged, character for character.
    """
    if not text.strip():
        return text, None, ()
    groups = {name: dict(values) for name, values in parse_fortran_namelist(text).items()}
    nhyd = {k.casefold(): k for k in groups.get('nhyd_model', {})}
    has_numerics = any(k.casefold() == 'config_numerics' for values in groups.values() for k in values)
    if not has_numerics and not any(key in nhyd for key in _MODE_KEYS):
        return text, None, ()
    mode = groups['nhyd_model'].pop(nhyd['config_physics_mode'], None) if 'config_physics_mode' in nhyd else None
    skip = groups['nhyd_model'].pop(nhyd['config_physics_correct_skip'], '') \
        if 'config_physics_correct_skip' in nhyd else ''
    for values in groups.values():
        for key in list(values):
            if key.casefold() == 'config_numerics':
                values.pop(key)
    if not isinstance(mode, (str, type(None))) or not isinstance(skip, str):
        raise ValueError('config_physics_mode and config_physics_correct_skip are character options')
    text = '\n'.join('&'+name+'\n'+'\n'.join(f'{k}={_literal(v)}' for k,v in values.items())+'\n/'
                     for name,values in groups.items())
    return text, None if mode is None else mode.strip(), tuple(s.strip() for s in skip.split(',') if s.strip())


def requested_physics_mode(arguments: Any, namelist_mode: str | None,
                           namelist_skip: tuple[str, ...]) -> tuple[str | None, tuple[str, ...]]:
    """The mode a run names (``--physics-mode`` or the namelist's config_physics_mode; None when neither
    does) and its skip list (``--physics-correct-skip``, repeatable or comma-separated, plus the namelist's)."""
    cli_mode = getattr(arguments, 'physics_mode', None)
    if cli_mode is not None and namelist_mode is not None and cli_mode != namelist_mode:
        raise ValueError(f'--physics-mode {cli_mode} and the namelist config_physics_mode={namelist_mode!r} '
                         'name two modes for one run')
    cli_skip = tuple(s.strip() for value in getattr(arguments, 'physics_correct_skip', None) or ()
                     for s in str(value).split(',') if s.strip())
    return (cli_mode if cli_mode is not None else namelist_mode,
            tuple(dict.fromkeys((*namelist_skip, *cli_skip))))


def require_rrfs_library(arguments):
    from .resources_build import RRFS_REFUSAL, resolve_resource
    supplied = getattr(arguments, 'native_physics_library', None) or os.environ.get('MPAS_NATIVE_LIBRARY')
    if not supplied:
        try:
            resolve_resource('native-library-rrfs')
        except ValueError:
            raise ValueError(RRFS_REFUSAL) from None


def native_route(arguments: Any, *, default_mode: str = 'exact'):
    """Table-selected explicit suite; mpas_atmphys_control.F:92-332.

    ``default_mode`` is the mode a resolved MPAS suite route takes when the run names none.  The library
    default is exact (replay and reproduction callers are unchanged); the forecast door passes correct
    (owner ruling 2026-10-13: correct mode is the standard; coordinator 2026-10-14: the default is correct).
    """
    path = getattr(arguments, 'physics_namelist', None)
    text = path.read_text() if path else ''
    text, namelist_mode, namelist_skip = split_physics_mode(text)
    named_mode, correct_skip = requested_physics_mode(arguments, namelist_mode, namelist_skip)
    physics_mode = named_mode if named_mode is not None else default_mode
    groups = parse_fortran_namelist(text) if text.strip() else {}
    supplied = getattr(arguments, 'physics_suite', None)
    if getattr(arguments, 'physics_backend', None) == 'mpasphys':
        # The public backend selector names the MPAS suite route, rather than a
        # column-provider row. Without a suite, use MPAS's Registry default
        # (bundled core_atmosphere/Registry.xml:2299). Keep an explicit suite.
        supplied = supplied or groups.get('physics', {}).get('config_physics_suite') or 'mesoscale_reference'
    table = load_table()
    providers = [(key,row) for key,row in table.items()
                 if isinstance(row,Mapping) and (row.get('backend')=='mpasphys' or row.get('backend_row')==key)
                 and 'suite_routes' in row]
    routes = {alias for _,row in providers for alias in row['suite_routes']}
    selected = supplied or groups.get('physics',{}).get('config_physics_suite')
    matches = [(key,row) for key,row in providers if selected in row['suite_routes']]
    if len(matches)>1:
        raise ValueError('native physics route is ambiguous in the suites table')
    provider = import_module(matches[0][1].get('driver_module',
                'hexcore.mpasphys.driver_v842')) if matches else None
    # --physics-default-backend replaces the route's table default (hex-cuda on the v8.4.2 suites) for every
    # slot that names no backend of its own; native-cpu there is the native referee.
    default_backend = getattr(arguments, 'physics_default_backend', None)
    route_namelist = (partial(provider.route_namelist, default_backend=default_backend)
                      if provider is not None else lambda text, **_: None)
    if supplied in routes and provider is not None:
        # An explicit provider alias may select an unchanged native namelist.
        # Compare table-resolved suite identities before retaining its bytes.
        # It takes the route's table default like the namelist route (it took the library's native-cpu
        # default before, so --physics-suite NAME ran every slot on the CPU).
        route_namelist = partial(provider.resolve_namelist,
                                 default_backend=default_backend or matches[0][1]['default_backend'])
        existing = groups.get('physics', {}).get('config_physics_suite')
        def native_suite(name):
            resolved=provider.resolve_namelist(f'&physics\nconfig_physics_suite={name!r}\n/\n')
            return getattr(resolved,'native_suite',resolved.suite)
        if existing and native_suite(existing) != native_suite(supplied):
            raise ValueError('command-line and namelist physics suites differ')
        if existing is None:
            text += f'\n&physics\nconfig_physics_suite={supplied!r}\n/\n'
    mode = dict(physics_mode=physics_mode, correct_skip=correct_skip)
    config = route_namelist(text, **mode) if text.strip() else None
    # A run with no MPAS suite route goes to the ArWen door, which reads the same --physics-mode and
    # --physics-correct-skip (hexcore.correct_mode, lane H1): its corrections are the engine's (WB4, WB5),
    # and one the engine cannot apply is a named gap in that run's receipt, never a refusal.  Only a mode
    # written in the namelist's &nhyd_model is refused there, because the ArWen door never reads it.
    if config is None and namelist_mode not in (None, 'exact'):
        raise ValueError('config_physics_mode in the namelist applies to an explicit MPAS suite route '
                         '(--physics-suite or config_physics_suite); this run has none.  Name the mode with '
                         '--physics-mode, which the ArWen route reads')
    if config is None and namelist_skip:
        raise ValueError('config_physics_correct_skip in the namelist applies to an explicit MPAS suite '
                         'route; this run has none.  Name the skips with --physics-correct-skip')
    if config is None and default_backend is not None:
        # THE BREAKAGE THIS PREVENTS: a run naming the MPAS slot backend (the native referee, say) with no
        # MPAS suite route would run ArWen's physics while its author believes it ran MPAS's.
        raise ValueError('--physics-default-backend applies to an explicit MPAS suite route (--physics-suite '
                         'or config_physics_suite); this run has none')
    if config is not None and getattr(arguments, 'physics', None):
        # Reconstruct one native physics group so parse_fortran_namelist never
        # discards a repeated group. WOOF-only selectors have no Registry row.
        groups = {name:dict(options) for name,options in parse_fortran_namelist(text).items()}
        for override in arguments.physics:
            key, separator, value = override.partition('=')
            if not separator:
                raise ValueError(f'physics override requires KEY=VALUE: {override}')
            spec = matches[0][1]
            backend_options = {row['backend_selector'] for row in spec['slots'].values()}
            backend_options.update(row['backend_selector'] for row in spec.get('services', {}).values())
            option = key.strip().casefold()
            registry_groups = spec.get('registry_groups',{})
            owners = [name for name,keys in registry_groups.items() if option in keys]
            group = (spec['backend_namelist_group'] if option in backend_options
                     else owners[0] if owners else 'physics')
            parsed = parse_fortran_namelist(f'&{group}\n{key}={value}\n/')[group]
            groups.setdefault(group, {}).update(parsed)
        text = '\n'.join('&'+name+'\n'+'\n'.join(f'{k}={_literal(v)}' for k,v in values.items())+'\n/'
                         for name,values in groups.items())
        config = route_namelist(text, **mode)
    return config, text


def bind_door_iau(arguments: Any, text: str) -> str:
    """The door's IAU flags on the explicit suite route, as MPAS's own inputs.

    ``--iau-increment FILE`` (or ``--iau-analysis`` with the run's --init as the background, or an analysis
    bundle with --iau on) and ``--iau-window-seconds`` become the namelist's &iau group
    (config_IAU_option='on', config_IAU_window_length_s) and the iau stream's input file, so the run reads
    and applies the increment exactly as an MPAS namelist that asks for IAU does (runtime_v842.SuiteIau).
    Returns the namelist text with that &iau group.
    """
    window = float(arguments.iau_window_seconds)
    if not window > 0:
        raise ValueError('--iau-window-seconds must be positive; a zero window is direct insertion')
    if getattr(arguments, 'iau_route', None) == 'hex':
        # Breakage: hexcore.iau_forcing's coupled increment is an ArWen-route executor hook; on this route the
        # flag would be accepted and never used.
        raise ValueError("--iau-route hex adds a coupled increment on the ArWen route's executors; the explicit "
                         "MPAS suite route applies MPAS's own IAU (atm_add_tend_anal_incr).  Drop --iau-route or "
                         "name native")
    groups = {name: dict(values) for name, values in parse_fortran_namelist(text).items()} if text.strip() else {}
    existing = {key.casefold(): value for key, value in groups.get('iau', {}).items()}
    if str(existing.get('config_iau_option', 'off')).strip().lower() != 'off':
        raise ValueError("the namelist's &iau and --iau-window-seconds both name an IAU window; give it once")
    increment = getattr(arguments, 'iau_increment', None)
    if increment is None:
        analysis = getattr(arguments, 'iau_analysis', None)
        if analysis is None:
            raise ValueError('--iau-window-seconds needs --iau-increment, or --iau-analysis to form one')
        if not Path(analysis).expanduser().is_file():
            raise ValueError(f'--iau-analysis {analysis} is not a file')
        from ..analysis_bundle import write_increment
        work = Path(str(Path(arguments.out).expanduser().resolve()) + '.iau-work')
        work.mkdir(parents=True, exist_ok=False)
        increment = work / 'iau-increment.nc'
        write_increment(Path(analysis).expanduser().resolve(), Path(arguments.init).expanduser().resolve(), increment)
    if not Path(increment).expanduser().is_file():
        raise ValueError(f'--iau-increment {increment} is not a file')
    bindings = list(getattr(arguments, 'stream_input', None) or ())
    if any(str(binding).partition('=')[0] == 'iau' for binding in bindings):
        raise ValueError('--stream-input iau= and the IAU flags both name the increment file; give it once')
    arguments.stream_input = [*bindings, f'iau={Path(increment).expanduser().resolve()}']
    groups['iau'] = {'config_IAU_option': 'on', 'config_IAU_window_length_s': window}
    return '\n'.join('&'+name+'\n'+'\n'.join(f'{k}={_literal(v)}' for k,v in values.items())+'\n/'
                     for name,values in groups.items())


def _literal(value: Any) -> str:
    """Namelist values preserve Registry kinds; mpas_atmphys_control.F:92-332."""
    if isinstance(value, bool):
        return '.true.' if value else '.false.'
    return repr(value)


def set_namelist_options(text: str, options: Mapping[str, Mapping[str, Any]]) -> str:
    """Set namelist options, keeping every other record (rebuilt as native_route rebuilds overrides)."""
    groups = {name: dict(values) for name, values in parse_fortran_namelist(text).items()} if text.strip() else {}
    for group, values in options.items():
        target = groups.setdefault(group, {})
        for key, value in values.items():
            for existing in [k for k in target if k.casefold() == key.casefold()]:
                del target[existing]
            target[key] = value
    return '\n'.join('&'+name+'\n'+'\n'.join(f'{k}={_literal(v)}' for k,v in values.items())+'\n/'
                     for name,values in groups.items())+'\n'


def namelist_option(text: str, group: str, key: str):
    """One namelist option (case-insensitive), or None."""
    if not text.strip():
        return None
    values = {k.casefold(): v for k, v in dict(parse_fortran_namelist(text).get(group, {})).items()}
    return values.get(key.casefold())


def apply_start_time(text: str, start_time: str | None) -> str:
    """``--start-time`` on the MPAS route names the run's config_start_time.

    THE BREAKAGE THIS PREVENTS: the MPAS route read its start only from the namelist, so a run (or a moving
    case's ``physics_args``) naming a suite without a namelist start ran silently from the Registry default,
    2014-09-10_00:00:00: wrong sun, wrong ozone month, wrong output times.  A namelist naming a different
    start than ``--start-time`` is refused.  A run without ``--start-time`` is unchanged.
    """
    if not start_time:
        return text
    named = namelist_option(text, 'nhyd_model', 'config_start_time')
    if named is not None:
        if str(named).strip() != start_time:
            raise ValueError(f'--start-time {start_time} and the namelist config_start_time {named} name two '
                             'starts for one run')
        return text
    return set_namelist_options(text, {'nhyd_model': {'config_start_time': start_time}})


def bind_init_start(text: str, init: Any) -> str:
    """The run's config_start_time from the init's own ``xtime`` when nothing else names one.

    THE BREAKAGE THIS PREVENTS: a suite run with no namelist start and no ``--start-time``
    (``gpuwm-hex forecast --physics-suite convection_permitting`` from an install) ran from MPAS's Registry
    default start, 2010-10-23_00:00:00: output named ``output.2010-10-23_00.00.00.nc``, the sun and the ozone
    and aerosol months of that date (seen 2026-10-16, hexmx proof, x1.2562 2023-03-31 12Z init).  The init is
    the authority for the run's start, as the door's help says; a namelist start or ``--start-time`` still
    decides, and a restart run (config_do_restart) keeps MPAS's own restart timestamp rule.
    """
    if namelist_option(text, 'nhyd_model', 'config_start_time') is not None:
        return text
    restart = namelist_option(text, 'restart', 'config_do_restart')
    if restart is True or str(restart).strip().lower() in ('.true.', 'true', 't'):
        return text
    if init is None:
        return text
    from ..lbc import _open_lbc, _read_xtime
    path = Path(init)
    try:
        valid = _read_xtime(_open_lbc(path), path)
    except Exception as error:
        raise ValueError(f'the init {path} carries no readable xtime, so the run has no start time; name '
                         f'--start-time or a namelist config_start_time ({error})') from error
    start = valid.strftime('%Y-%m-%d_%H:%M:%S')
    import sys
    print(f'gpuwm-hex: MPAS suite start {start} (the init\'s xtime)', file=sys.stderr, flush=True)
    return set_namelist_options(text, {'nhyd_model': {'config_start_time': start}})


def _mpas_interval(seconds: float) -> str:
    """Seconds as an MPAS interval, D_HH:MM:SS."""
    whole = int(round(seconds))
    if whole <= 0 or abs(seconds - whole) > 1e-6:
        raise ValueError(f'an MPAS stream interval must be a whole positive number of seconds, not {seconds}')
    days, rest = divmod(whole, 86400)
    return f'{days}_{rest // 3600:02d}:{rest % 3600 // 60:02d}:{rest % 60:02d}'


_HEADER_KEYS = ('type', 'filename_template', 'filename_interval', 'reference_time', 'record_interval',
                'clobber_mode', 'precision', 'io_type', 'packages', 'input_interval', 'output_interval')


def _in_defaults(element: Any) -> int:
    """streams_gen's is_structure_writable for the build's one key, in_defaults=true: 1, 0, or -1 (absent)."""
    value = element.get('in_defaults')
    if value is None:
        return -1
    return 1 if 'true' in value.split(';') else 0


def write_default_streams(layout: Any, path: Path, *, history_minutes: float | None = None,
                          checkpoint_hours: Any = None) -> Path:
    """MPAS's own default streams.atmosphere and stream_list.atmosphere.* files, from the Registry this run decoded.

    THE BREAKAGE THIS PREVENTS: the MPAS suite route refused every run that named no streams.atmosphere
    ("requires the actual streams.atmosphere (--streams)") and every run whose streams file named
    stream_list.atmosphere.{output,diagnostics,surface,diag_ugwp} files the install did not ship, so
    ``gpuwm-hex forecast --physics-suite convection_permitting`` and ``gpuwm-hex global`` with an MPAS suite
    could not run from an install at all: only hand-made decks carrying a native run directory's files could
    (seen 2026-10-16, image proofs).  MPAS's build writes both from the preprocessed Registry
    (src/core_atmosphere/Makefile:38, ``streams_gen Registry_processed.xml streams.atmosphere
    stream_list.atmosphere. listed in_defaults=true``; src/tools/input_gen/streams_gen.c); this does the same,
    in the same order: every stream ``in_defaults`` admits, a separate_file stream naming its
    ``stream_list.atmosphere.<name>`` (var_arrays then vars, one per line) written beside the streams file,
    a single_file stream carrying its members inline.  The run's own choices are then applied: output and
    diagnostics every ``history_minutes`` (default: the Registry's own intervals), each time to its own file;
    restart only at ``checkpoint_hours`` (a single interval), else never.  A run that names its streams file
    is unchanged.
    """
    root = ET.Element('streams')
    history = _mpas_interval(float(history_minutes) * 60.0) if history_minutes else None
    restart = 'none'
    if checkpoint_hours:
        values = [v for v in str(checkpoint_hours).split(',') if v.strip()]
        if len(values) == 1:
            restart = _mpas_interval(float(values[0]) * 3600.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    for declared in layout.root.findall('./streams/stream'):
        name = declared.get('name')
        immutable = declared.get('immutable') == 'true'
        writable = _in_defaults(declared) != 0
        if immutable:
            if not writable:
                continue
            members = []
        else:
            members = [m for tag in ('stream', 'var_struct', 'var_array', 'var') for m in declared.findall(tag)
                       if _in_defaults(m) == 1 or (_in_defaults(m) == -1 and writable)]
            if not writable and not members:
                continue
        element = ET.SubElement(root, 'immutable_stream' if immutable else 'stream', {'name': name})
        for key in _HEADER_KEYS:
            if declared.get(key) is not None:
                element.set(key, declared.get(key))
        types = declared.get('type', '').split(';')
        if name in ('output', 'diagnostics') and history:
            element.set('output_interval', history)
        if name == 'restart':
            element.set('output_interval', restart)
        if 'output' in types and element.get('output_interval', 'none') != 'none' \
                and declared.get('filename_interval') is None:
            element.set('filename_interval', 'output_interval')
        if immutable:
            continue
        separate = declared.get('runtime_format') == 'separate_file'
        listed = []
        if separate:
            list_name = f'stream_list.atmosphere.{name}'
            ET.SubElement(element, 'file', {'name': list_name})
        for member in members:
            if separate and member.tag in ('var_array', 'var'):
                listed.append(member.get('name'))
                continue
            attributes = {'name': member.get('name')}
            if member.get('packages') is not None:
                attributes['packages'] = member.get('packages')
            ET.SubElement(element, member.tag, attributes)
        if separate:
            (path.parent / list_name).write_text(''.join(f'{field}\n' for field in listed))
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding='unicode') + '\n')
    return path


def read_streams(path: Path, layout: Any, list_directory: Path | None = None, *, tree=None) -> tuple[Stream, ...]:
    """Resolve Registry/file lists and packages; Registry.xml:498-1351."""
    active = {r['name']:r for r in layout.fields}
    declared = {s.get('name'):s for s in layout.root.findall('./streams/stream')}
    # MPAS recursive stream expansion: Registry.xml:498-1351.
    def members(element, seen=(), order=None):
        result = []
        # Immutable Registry streams are compiled by member kind, rather than
        # XML declaration order: MPAS gen_inc.c:2095-2188.
        children = list(element) if order is None else [child for tag in order for child in element.findall(tag)]
        for child in children:
            if not layout.enabled(child):
                continue
            name = child.get('name')
            if child.tag in ('var','var_array'):
                if name in active:
                    result.append(name)
            elif child.tag=='var_struct':
                # Native allocates variable arrays before ordinary fields, even
                # when their XML declarations are interleaved: gen_inc.c:1944-1952.
                rows = [r for r in layout.fields if r['pool'] == name]
                result.extend(r['name'] for arrays in (True, False) for r in rows
                              if bool(r['components']) == arrays)
            elif child.tag=='stream':
                if name in seen:
                    raise ValueError(f'cyclic native stream {name}')
                result.extend(members(declared[name], (*seen,name), ('var', 'var_array', 'var_struct')))
            elif child.tag=='file':
                member_path=(list_directory or path.parent)/name
                for line in member_path.read_text().splitlines():
                    field = line.split('#',1)[0].strip()
                    if field:
                        result.append(field)
            else:
                raise ValueError(f'unknown MPAS stream member {child.tag}')
        return result
    streams=[]
    for element in (ET.parse(path).getroot() if tree is None else tree):
        if not layout.enabled(element):
            continue
        name=element.get('name')
        fields=(members(declared[name], (name,), ('stream', 'var_struct', 'var_array', 'var'))
                if element.tag == 'immutable_stream' else members(element))
        streams.append(Stream(name,tuple(element.get('type').split(';')),
                              element.get('filename_template'),element.get('input_interval','none'),
                              element.get('output_interval','none'),tuple(dict.fromkeys(fields)),
                              {'input_interval':element.get('input_interval','none'),
                               'output_interval':element.get('output_interval','none')}.get(
                                   element.get('filename_interval','none'),
                                   element.get('filename_interval','none')),
                              options=dict(element.attrib)))
    return tuple(streams)


def run_forecast(arguments: Any, config: Any, text: str) -> int:
    """Hydrate Registry pools for resident physics; core.F:589-596."""
    pin_set = getattr(config, 'pin_set', None) or 'v8.4.2'
    if pin_set == 'rrfs':
        require_rrfs_library(arguments)
    reader = getattr(arguments, 'native_pool_reader', None) or os.environ.get('MPAS_NATIVE_POOL_READER')
    source = getattr(arguments, 'mpas_source', None) or os.environ.get('MPAS_SOURCE')
    from .runtime_v842 import framework_only
    ported_framework = config is None or framework_only(config)
    if ported_framework:
        # Framework pools, clock and initialization are Python-owned. The
        # expanded pinned Registry ships with the wheel. No native library or
        # MPAS source resource is requested, including on a clean install.
        from .framework_backend_v842 import registry_source
        library = None
        source = source or registry_source()
        if not reader:
            from .resources_build import ensure_route_resources
            reader = ensure_route_resources(('pool-reader',), pin_set=pin_set)['pool-reader']
    else:
        # The separate RRFS fork retains its own provider and pin set.
        library = getattr(arguments, 'native_physics_library', None) or os.environ.get('MPAS_NATIVE_LIBRARY')
        from .resources_build import ROUTE_KEYS, ensure_route_resources
        keys = ROUTE_KEYS[pin_set]
        unnamed = tuple(key for key, value in ((keys['native-library'], library), ('pool-reader', reader),
                                               (keys['mpas-source'], source)) if not value)
        if unnamed:
            installed = ensure_route_resources(unnamed, pin_set=pin_set)
            library = library or installed.get(keys['native-library'])
            reader = reader or installed.get('pool-reader')
            source = source or installed.get(keys['mpas-source'])
    required = {'native-pool-reader': reader, 'mpas-source': source,
                'init': getattr(arguments, 'init', None)}
    if not ported_framework:
        required['native-physics-library'] = library
    for option, value in required.items():
        if not value:
            raise ValueError(f'MPAS physics route requires --{option}: its provider is not bound')
        if not Path(value).exists():
            raise ValueError(f'MPAS physics route --{option} does not exist: {value}')
    if config is not None and library is not None:
        from ..licence_gate import native_library_refusal
        refusal = native_library_refusal(config, Path(library))
        if refusal is not None:
            raise refusal
    namelist = getattr(arguments,'physics_namelist',None)
    # A run that names no streams file takes the namelist's sibling when there is one, else MPAS's own default
    # written from the Registry once the pools are decoded (write_default_streams).
    streams_path = getattr(arguments,'streams',None)
    if streams_path is None and namelist is not None and namelist.with_name('streams.atmosphere').is_file():
        streams_path = namelist.with_name('streams.atmosphere')
    if streams_path is not None and not Path(streams_path).is_file():
        raise ValueError(f'MPAS native route: the named streams.atmosphere does not exist: {streams_path}')
    out = getattr(arguments,'out',None)
    if out is None:
        raise ValueError('MPAS native route requires --out for stream products and the receipt')
    out = Path(out).resolve()
    scratch = getattr(arguments,'scratch',None) or out/'scratch'
    scratch = Path(scratch).resolve()
    sources = [Path(v).resolve() for k in ('grid','static','init','orographic_statistics')
               if (v:=getattr(arguments,k,None)) is not None]
    bindings = stream_input_bindings(getattr(arguments,'stream_input',None))
    row = load_table()[getattr(config,'table_key','mpasphys_v842')]
    table_bindings = {}
    selector = row.get('native_table_stream_selector')
    if selector is not None:
        module, operation = selector.split(':', 1)
        roots = [Path(source).resolve()/'src/core_atmosphere/physics/physics_wrf/files']
        extra = getattr(arguments,'native_physics_tables',None) or []
        keyword = row.get('native_table_override_keyword')
        callback = getattr(import_module(module), operation)
        table_bindings = (callback(config, roots, **{keyword: extra}) if keyword
                          else callback(config, [*roots, *extra]))
    # Timed surface input is first read in the forecast loop, after startup
    # output; mpas_atm_core.F:198-210,655-724,779-806. Preserve init SST/xice
    # during pool hydration and install timed records only on their alarms.
    startup_bindings = (path for name, path in bindings.items()
                        if name != 'surface' or not config.options['config_sst_update'])
    sources = list(dict.fromkeys((*sources,*table_bindings.values(),*startup_bindings)))
    text = apply_start_time(text, getattr(arguments, 'start_time', None))
    from .checkpoint_v842 import resolve_route
    moving, text = resolve_route(arguments, text)
    text = bind_init_start(text, getattr(arguments, 'init', None))
    from ..stage_timing import stage
    timings = []
    out.mkdir(parents=True, exist_ok=True)
    def save_timings():
        (out/'stage-timings.json').write_text(json.dumps({'stages':timings},indent=2)+'\n')
    try:
        with stage(timings, 'native-pool-decode', parallelism={'device':'CPU','threads':1},
                   timeline='prep.native.pool_decode'):
            layout = decode_registry_pools(Path(reader).resolve(),Path(source).resolve(),config,text,sources,
                                           scratch/'native-pools')
    finally:
        save_timings()
    if moving is not None:
        moving.mark_restart(layout)
    if streams_path is None:
        streams_path = write_default_streams(layout, scratch/'streams.atmosphere',
                                             history_minutes=getattr(arguments,'history_every_minutes',None),
                                             checkpoint_hours=None)
    parser=load_table()[getattr(config,'table_key','mpasphys_v842')].get('streams_parser')
    tree=None
    if parser is not None:
        module,callable_name=parser.split(':')
        tree=getattr(import_module(module),callable_name)(streams_path,library)
    streams=read_streams(streams_path,layout,getattr(arguments,'stream_list_directory',None),tree=tree)
    defaults = {'input':Path(arguments.init).resolve()}
    if getattr(arguments,'orographic_statistics',None):
        defaults['ugwp_oro_data_in'] = Path(arguments.orographic_statistics).resolve()
    active = {s.name for s in streams if 'input' in s.directions}
    defaults = {name:path for name,path in defaults.items() if name in active}
    streams = bind_stream_inputs(streams,{**defaults,**table_bindings,**bindings})
    start=layout.configs['config_start_time']
    dt=Fraction(str(layout.configs['config_dt']))
    selected_dt=getattr(arguments,'dt',None)
    if selected_dt not in (None,'auto','row'):
        dt=Fraction(str(selected_dt))
    if dt<=0:
        raise ValueError('native physics needs a positive config_dt')
    hours=getattr(arguments,'hours',None)
    duration=Fraction(str(hours))*3600 if hours is not None else interval_seconds(layout.configs['config_run_duration'])
    duration = integral_duration(duration, dt)
    if callable(getattr(moving, 'resolve_schedule', None)):
        duration = moving.resolve_schedule(duration, dt, streams)
    stop=(datetime.strptime(start,'%Y-%m-%d_%H:%M:%S')+timedelta(seconds=float(duration))).strftime('%Y-%m-%d_%H:%M:%S')
    out.mkdir(parents=True,exist_ok=True)
    from .. import numerics as _numerics
    plan=dict(schema='mpas-native-door-v1',physics=config.receipt(),numerics=_numerics.receipt_block(),
              start=start,stop=stop,dt=str(dt),
              grid=None if getattr(arguments,'grid',None) is None else str(Path(arguments.grid).resolve()),
              init=str(Path(arguments.init).resolve()),
              steps=int(duration/dt),streams=[s.__dict__ for s in streams],
              native_library=None if library is None else str(Path(library).resolve()),
              framework='python' if library is None else 'native-referee',
              rust_pool_reader=str(Path(reader).resolve()),
              pool_manifest=str(scratch/'native-pools/manifest.json'),identity_checked=False)
    (out/'native-physics-request.json').write_text(json.dumps(plan,indent=2)+'\n')
    if getattr(arguments,'preflight',False):
        print(json.dumps(dict(status='native-pool-preflight',identity_checked=False,request=plan)))
        return 0
    from .features_v842 import run_native_dynamics
    parallelism = {'device':'GPU plus Python framework' if library is None else 'GPU plus native CPU','cards':1,
                   'requested_omp_threads':int(os.environ.get('OMP_NUM_THREADS','1'))}
    if getattr(arguments, 'samehost_devices', None):
        from ..drivers.samehost_forecast_v841 import device_list
        parallelism.update(device='MPI GPU ranks plus native CPU',
                           cards=len(device_list(arguments.samehost_devices)))
    try:
        with stage(timings, 'native-forecast-runtime', parallelism=parallelism):
            rc = run_native_dynamics(arguments, config, layout, scratch/'native-pools', streams,
                                     None if library is None else Path(library).resolve(),
                                     start, stop, float(dt), int(duration/dt), out,
                                     moving=moving)
        if rc == 0:
            from ..history_writer import mark_closed_frame_durable
            from ..products_door import native_output_frames
            with stage(timings, 'native-history-durability', parallelism={'device':'CPU','threads':1},
                       timeline='post.native.history_durability'):
                for frame in native_output_frames(out):
                    mark_closed_frame_durable(frame)
        return rc
    finally:
        save_timings()
