"""Compile exact test images on CPU and replay them without invoking a compiler.

The test loader checks the complete CUDA source, strict compiler options,
architecture and image SHA before loading the real CUDA module. It replaces
only the compilation step in the unchanged production launcher. Tables and
candidate outputs follow the regular Rust/native/GPU replay protocols.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

from hexcore.mpasphys import thompson_v842 as port
try:
    from .replay import engine_math_source
except ImportError:
    from replay import engine_math_source


def source_key(source):
    return hashlib.sha256(source.encode()).hexdigest()


def compile_images(engine, output, architecture, levels=(), native_log10_only=False):
    from cupy.cuda import compiler
    try:
        from .helper_replay import helper_source
        from .effectrad_replay import effectrad_source
    except ImportError:
        from helper_replay import helper_source
        from effectrad_replay import effectrad_source
    output.mkdir(parents=True, exist_ok=True)
    for promoted in ((False,) if native_log10_only else (False, True)):
        maths = engine_math_source(engine.resolve(), promoted, native_log10_only)
        source = port.cuda_source(device_math_source=maths)
        texts = [port.cuda_source(device_math_source=maths, max_levels=n)
                 for n in sorted(set((*levels, port.MAX_LEVELS)))]
        texts.extend((source + helper_source(), source + effectrad_source()))
        for text in texts:
            key = source_key(text)
            certificate = output / (key + '.json')
            image_path = output / (key + '.cubin')
            if certificate.is_file() and image_path.is_file():
                metadata = json.loads(certificate.read_text())
                if (metadata['source_sha256'] == key
                        and metadata['options'] == list(port.COMPILE_OPTIONS)
                        and metadata['architecture'] == architecture
                        and metadata.get('production_sha256') == hashlib.sha256(Path(port.__file__).read_bytes()).hexdigest()
                        and metadata['cubin_sha256'] == hashlib.sha256(image_path.read_bytes()).hexdigest()):
                    print(key + ' authenticated existing image', flush=True)
                    continue
            image, _ = compiler.compile_using_nvrtc(text, options=port.COMPILE_OPTIONS,
                                                    arch=architecture, cache_in_memory=True)
            (output / (key + '.cubin')).write_bytes(image)
            (output / (key + '.json')).write_text(json.dumps({
                'acceptance': False, 'source_sha256': key,
                'options': list(port.COMPILE_OPTIONS), 'architecture': architecture,
                'cubin_sha256': hashlib.sha256(image).hexdigest(),
                'production_sha256': hashlib.sha256(Path(port.__file__).read_bytes()).hexdigest(),
                'diagnostic_log10_narrowing': promoted, 'shared_glibc243_log10': native_log10_only}, indent=2) + '\n')
            print(key, flush=True)


def install_loader(directory):
    def load(source):
        prefix = directory / source_key(source)
        metadata = json.loads(prefix.with_suffix('.json').read_text())
        image = prefix.with_suffix('.cubin').read_bytes()
        assert metadata['source_sha256'] == source_key(source)
        assert metadata['options'] == list(port.COMPILE_OPTIONS)
        assert metadata['cubin_sha256'] == hashlib.sha256(image).hexdigest()
        assert metadata['production_sha256'] == hashlib.sha256(Path(port.__file__).read_bytes()).hexdigest()
        import cupy as cp
        from cupy.cuda import function
        assert metadata['architecture'] == str(cp.cuda.Device().compute_capability)
        module = function.Module()
        module.load(image)
        return module
    port._module.cache_clear()
    port.compile_module = load
    return load


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile', type=Path)
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--architecture')
    parser.add_argument('--levels', type=int, action='append', default=[])
    parser.add_argument('--native-log10-only', action='store_true')
    parser.add_argument('--load', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.compile:
        if not args.engine or not args.architecture or not args.architecture.isdecimal():
            parser.error('--compile requires --engine and numeric --architecture')
        compile_images(args.engine, args.compile, args.architecture, args.levels, args.native_log10_only)
    else:
        if not args.load or not args.command:parser.error('--load requires a replay script and arguments')
        install_loader(args.load.resolve())
        command = args.command[1:] if args.command[0] == '--' else args.command
        sys.argv = command
        runpy.run_path(command[0], run_name='__main__')
