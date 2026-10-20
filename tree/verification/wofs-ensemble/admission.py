"""Pure CPU arithmetic on the generated count, including rollback charge."""
import os
from pathlib import Path
import json
from hexcore.device_admission import KNOWN_CARDS, model_for_card
w=Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))
cells=json.loads((w/'evidence/mesh.json').read_text())['emitted']['n_cells']
variants={}
for rollback in (False,True):
    model=model_for_card(KNOWN_CARDS['96gib-188sm'],levels=55,rollback_armed=rollback)
    variants[str(rollback)]=dict(required_mib=model.required_bytes(cells)/2**20,
        predicted_peak_mib=model.predict_bytes(cells)/2**20,provenance=model.provenance,
        rollback_armed=rollback)
result=dict(cells=cells,levels=55,card='96gib-188sm',variants=variants)
(w/'evidence/cpu-admission.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
