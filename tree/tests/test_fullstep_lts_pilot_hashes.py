"""The identity instrument must hash numeric arrays, not type object pointers."""
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import numpy as np


def test_array_digest_ignores_dtype_type_and_sees_changed_data():
    path=Path(__file__).parents[1]/'tools/census_fullstep_lts.py'
    spec=importlib.util.spec_from_file_location('census_pilot',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    @dataclass
    class State:
        rho: object
        dtype: object=np.float32
    value=np.array([[1.,2.]],dtype=np.float32)
    before=module.digest_arrays(State(value))
    assert before=={'rho':hashlib.sha256(value.tobytes()).hexdigest()}
    assert module.digest_arrays(State(value.copy()))==before
    value[0,0]=3
    assert module.digest_arrays(State(value))!=before
