"""Emit the complete owner reconciliation for the nested ozone initializer.

Only an exact proposal is written. The native scientific sources stay pristine.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    proposed, originals = {}, {}
    def read(name):
        text = (args.checkout / name).read_text()
        originals[name] = text
        return text
    def replace(text, old, new):
        assert text.count(old) == 1, old
        return text.replace(old, new)
    builder = 'tree/tools/native_physics_reference/backend/build.py'
    text = read(builder)
    text = replace(text, "    cobjs = []\n", """    cobjs = []
    # Framework callback only; MPAS mpas_atmphys_init.F:397. Scientific objects
    # remain pristine. --wrap routes this one nested call at the original site.
    ozone_bridge = work / 'native_ozone_initializer.c'
    cobjs.append('objects/native_ozone_initializer.o')
    lines += ['objects/native_ozone_initializer.o: ' + quote(str(ozone_bridge)) + ' compile-recipe.json',
              '\\tmpicc -O3 -fPIC -Wall -Wextra -Werror -c $< -o $@', '']
""")
    text = replace(text, "f'\\t{fc} -shared -Wl,-z,defs -Wl,-Bsymbolic-functions -o $@ '",
        "f'\\t{fc} -shared -Wl,-z,defs -Wl,-Bsymbolic-functions -Wl,--wrap=__mpas_atmphys_o3climatology_MOD_init_o3climatology -o $@ '")
    text = replace(text, "               'wrappers_sha256': hashlib.sha256(wrapper.read_bytes()).hexdigest(),",
        "               'ozone_initializer_bridge_sha256': hashlib.sha256(ozone_bridge.read_bytes()).hexdigest(),\n"
        "               'wrappers_sha256': hashlib.sha256(wrapper.read_bytes()).hexdigest(),")
    proposed[builder] = text
    generator = 'tree/tools/native_physics_reference/backend/generate.py'
    text = read(generator)
    bridge = Path(__file__).with_name('native_ozone_initializer_bridge.c').read_text()
    text = replace(text, "    for file, text in [('native_wrappers.f90', '\\n'.join(f) + '\\n'),",
        "    # MPAS mpas_atmphys_init.F:397; mpas_atmphys_o3climatology.F:59-194.\n"
        "    write_changed(output / 'native_ozone_initializer.c', " + repr(bridge) + ")\n"
        "    for file, text in [('native_wrappers.f90', '\\n'.join(f) + '\\n'),")
    proposed[generator] = text
    adapter = 'tree/src/hexcore/mpasphys/native_backend.py'
    text = read(adapter)
    helpers = '''    def set_ozone_initializer(self, callback):
        """Selected nested callback; mpas_atmphys_init.F:397; ozone.F:59-194."""
        with _LOCK:
            self._check()
            if callback is not None:
                for name in ('native_ozone_initializer_set', 'native_ozone_initializer_status',
                             'native_ozone_initializer_calls'):
                    if not hasattr(self.lib, name):
                        raise RuntimeError('native provider lacks the nested ozone callback ABI; prevents silently retaining native initialization for a selected CUDA initializer')
                self.lib.native_ozone_initializer_set.argtypes = [ct.c_void_p]
                self.lib.native_ozone_initializer_set.restype = None
                for name in ('native_ozone_initializer_status', 'native_ozone_initializer_calls'):
                    getattr(self.lib, name).argtypes = []
                    getattr(self.lib, name).restype = ct.c_int
            self._ozone_initializer = callback

    def _call_with_ozone_initializer(self, entry, values):
        """Route the original nested call; mpas_atmphys_init.F:397."""
        handles = dict(zip(self.abi['entries'][entry]['arguments'], values, strict=True))
        errors = []
        @ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p)
        def invoke(mesh, atm_input):
            try:
                if mesh != handles['mesh'] or atm_input != handles['atm_input']:
                    raise RuntimeError('nested ozone handles differ from native physics_init pools')
                self._ozone_initializer()
                return 0
            except BaseException as error:
                errors.append(error)
                return 1
        self.lib.native_ozone_initializer_set(ct.cast(invoke, ct.c_void_p))
        try:
            getattr(self.lib, 'native_' + entry)(*values)
            status = self.lib.native_ozone_initializer_status()
            calls = self.lib.native_ozone_initializer_calls()
        finally:
            self.lib.native_ozone_initializer_set(None)
        if errors:
            return errors[0]
        if status or calls != 1:
            return RuntimeError(f'native nested ozone callback status={status}, calls={calls}; selected initialization did not execute exactly once')
        return None

'''
    marker = '    # Native binding/orchestration authority: src/core_atmosphere/physics/mpas_atmphys_driver.F:209-384; src/core_atmosphere/physics/mpas_atmphys_todynamics.F:98-498.\n'
    text = replace(text, marker, helpers + marker)
    text = replace(text, "            getattr(self.lib, 'native_' + name)(*values)\n",
        "            initialization_error = None\n"
        "            if name == 'physics_init' and getattr(self, '_ozone_initializer', None) is not None:\n"
        "                initialization_error = self._call_with_ozone_initializer(name, values)\n"
        "            else:\n"
        "                getattr(self.lib, 'native_' + name)(*values)\n")
    text = replace(text, "            elif name == 'atmphys_finalize':\n                self._initialized = False\n",
        "            elif name == 'atmphys_finalize':\n                self._initialized = False\n"
        "            if initialization_error is not None:\n                raise initialization_error\n")
    proposed[adapter] = text
    driver = 'tree/src/hexcore/mpasphys/driver_v842.py'
    text = read(driver)
    text = replace(text, "        if masked:\n            self.native.config(\"config_ngw_scheme\", call.options[\"config_ngw_scheme\"])\n        try:\n",
        "        if masked:\n            self.native.config(\"config_ngw_scheme\", call.options[\"config_ngw_scheme\"])\n"
        "        initializer = self.external.get('init_o3climatology') if entry == 'physics_init' else None\n"
        "        if initializer is not None:\n            self.native.set_ozone_initializer(lambda: initializer(call))\n"
        "        try:\n")
    text = replace(text, "        finally:\n            if masked:\n                self.native.config(\"config_ngw_scheme\", original)\n",
        "        finally:\n            if initializer is not None:\n                self.native.set_ozone_initializer(None)\n"
        "            if masked:\n                self.native.config(\"config_ngw_scheme\", original)\n")
    proposed[driver] = text
    # Existing separate time/device patches are included verbatim in this
    # complete reconciliation; initializer descriptor joins the same callbacks.
    device = 'tree/src/hexcore/mpasphys/device_pools_v842.py'
    text = read(device)
    text = replace(text, '        if resource:\n', "        if resource and config.options.get(resource.get('enabled_option'), True):\n")
    text = replace(text, '        registry.register(slot,scheme,Backend.HEX_CUDA,invoke)\n', '''        registry.register(slot,scheme,Backend.HEX_CUDA,invoke)
        for operation, entry in descriptor.get('external_entries', {}).items():
            if not config.options.get(entry.get('enabled_option'), True):
                continue
            external_callback=getattr(module,entry['call'])
            external_descriptor=dict(entry,module=descriptor['module'])
            def external_invoke(call,callback=external_callback,descriptor=external_descriptor):
                """MPAS mpas_atmphys_o3climatology.F:59-194,234-327."""
                prepared=provider.begin(call,descriptor)
                result=callback(replace(call,pools=prepared))
                provider.commit()
                return result
            provider.session.hooks.external[operation]=external_invoke
''')
    proposed[device] = text
    table = 'tree/src/hexcore/data/profiles/mpas-physics-suites.v1.json'
    text = read(table)
    data = json.loads(text)
    for descriptor in data['mpasphys_v842']['slots']['cloudiness']['cuda_entries'].values():
        assert descriptor['module'] == 'radshared_v842'
        descriptor['resources'] = dict(module='radshared_v842', call='load_ozone_resources',
            attribute='radshared_v842_tables', enabled_option='config_o3climatology')
        descriptor['external_entries'] = {
            'init_o3climatology': dict(call='init_ozone_slot', bind='bind_native_ozone_initial_pools', enabled_option='config_o3climatology'),
            'o3climatology_from_MPAS': dict(call='ozone_slot', bind='bind_native_ozone_pools', enabled_option='config_o3climatology')}
    proposed[table] = json.dumps(data, indent=2) + '\n'
    patches = []
    for name, text in proposed.items():
        if name.endswith('.py'):
            ast.parse(text)
        patches.extend(difflib.unified_diff(originals[name].splitlines(True), text.splitlines(True),
            fromfile='a/' + name, tofile='b/' + name))
    args.output.write_text(''.join(patches))
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        schema='p21-complete-nested-ozone-owner-proposal/v1', applied=False,
        files={name: dict(original_sha256=hashlib.sha256(originals[name].encode()).hexdigest(),
            proposed_sha256=hashlib.sha256(text.encode()).hexdigest()) for name, text in proposed.items()},
        bridge_sha256=hashlib.sha256(bridge.encode()).hexdigest(),
        production_scientific_source_modified=False, forecast_acceptance=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
