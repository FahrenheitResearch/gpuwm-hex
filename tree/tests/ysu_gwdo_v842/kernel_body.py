"""Test build orchestration: extract the production CUDA body without importing GPU code."""
import ast
from pathlib import Path
import sys
module=ast.parse(Path(sys.argv[1]).read_text())
for statement in module.body:
    if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_KERNEL_BODY' for t in statement.targets):
        body = ast.literal_eval(statement.value).replace('@NLEV@','nlev')
        native_dx = body.replace('bl_ysu_gwdo_v842_f32(', 'bl_ysu_gwdo_v842_native_dx_f32(')
        native_dx = native_dx.replace('mpas_div(len_disp, gfk_pow(md, 0.25f))', 'md')
        Path(sys.argv[2]).write_text(body + '\n' + native_dx)
        break
else:
    raise RuntimeError('production CUDA body missing')
