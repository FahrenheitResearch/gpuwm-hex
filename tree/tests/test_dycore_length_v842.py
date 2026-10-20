"""Native core control replay; no forecast claim."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pytest
from hexcore.mixing_v842 import V842MixingConfig,resolve_config_len_disp


def test_nonpositive_length_matches_pristine_native_core():
    folder=os.environ.get("HEXCORE_D1_LENGTH_FIXTURE")
    if not folder:
        pytest.skip("set HEXCORE_D1_LENGTH_FIXTURE to compiled native length oracle")
    folder=Path(folder);data=(folder/"length.bin").read_bytes()
    before=np.frombuffer(data,"<f4",count=16).reshape(2,8,order="F")
    after=np.frombuffer(data,"<f4",count=16,offset=64).reshape(2,8,order="F")
    errors=np.frombuffer(data,"<i4",count=8,offset=128)
    rows=[]
    for i in range(8):
        configured,nominal=before[:,i]
        V842MixingConfig(config_len_disp=float(configured)).validate()
        if errors[i]:
            with pytest.raises(ValueError):resolve_config_len_disp(nominal,configured)
            rows.append({"case":i,"native_error":int(errors[i]),"port_error":True})
        else:
            actual=np.float32(resolve_config_len_disp(nominal,configured))
            expected=after[0,i]
            difference=int(actual.view(np.uint32)!=expected.view(np.uint32))
            rows.append({"case":i,"words":1,"different_words":difference})
            assert difference==0
    receipt=os.environ.get("HEXCORE_D1_LENGTH_RECEIPT")
    if receipt:
        Path(receipt).write_text(json.dumps({"scope":"strict unchanged native length-control block",
            "status":"PASS","source":"mpas_atm_core.F:168-201","comparisons":rows,
            "native_binary_sha256":hashlib.sha256((folder/"length-deck").read_bytes()).hexdigest(),
            "native_wrapper_sha256":hashlib.sha256((folder/"core_length.f90").read_bytes()).hexdigest(),
            "native_output_sha256":hashlib.sha256(data).hexdigest()},indent=2)+"\n")
