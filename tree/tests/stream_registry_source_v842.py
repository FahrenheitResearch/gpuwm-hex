"""Reproduce the pinned stream metadata for source-fixture checks.

This is a test fixture generator. The independent oracle remains MPAS's own
compiled C registry tool, whose emitted Fortran is checked separately.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib, json
root=None
pins={}
def preprocess(path, physics):
    pins[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    active=[True]; out=[]
    for line in path.read_text().splitlines():
        s=line.strip()
        if s.startswith(('#ifdef ', '#ifndef ')):
            truth=(s.split()[1]=='DO_PHYSICS' and physics)
            active.append(active[-1] and (not truth if s.startswith('#ifndef') else truth))
        elif s=='#endif': active.pop()
        elif s=='#else': raise RuntimeError('unsupported conditional')
        elif s.startswith('#include'):
            if active[-1]: out.append(preprocess(path.parent/s.split('"')[1],physics))
        elif not s.startswith('#') and active[-1]: out.append(line)
    assert len(active)==1
    return '\n'.join(out)
def extract(physics):
    tree=ET.fromstring(preprocess(root/'Registry.xml',physics))
    fields={}; groups={}
    def visit(el, inherited, gates):
        att=el.attrib
        pkg=att.get('packages','')
        gates=(*gates,pkg) if pkg else gates
        if el.tag=='var_array':
            inherited={k:att[k] for k in ('type','dimensions')}
            groups.setdefault(att['name'],[]).extend(n.attrib['name'] for n in el.iter('var'))
        if el.tag=='var_struct': groups.setdefault(att['name'],[]).extend(n.attrib['name'] for n in el.iter('var'))
        if el.tag=='var':
            a={**inherited,**att}; dims=a.get('dimensions','').split()
            if a['type']=='text': dims=['StrLen',*dims]
            ignored={'name','type','dimensions','persistence','packages','time_levs','name_in_code','array_group','default_value'}
            attrs={ {'description':'long_name','missing_value':'_FillValue'}.get(k,k):(int(v) if a['type'] in ('integer','logical') else float(v)) if k=='missing_value' else v for k,v in a.items() if k not in ignored}
            fields[a['name']]={'type':a['type'],'dimensions':dims,'attrs':attrs,'gates':gates}
        for child in el: visit(child,inherited,gates)
    for el in tree.findall('var_struct'): visit(el,{},())
    streams={}
    for stream in tree.find('streams'):
        streams[stream.attrib['name']]={'options':dict(stream.attrib),'members':[[c.tag,c.attrib['name'],c.attrib.get('packages','')] for c in stream]}
    return fields,{k:list(dict.fromkeys(v)) for k,v in groups.items()},streams,list(dict.fromkeys(n.attrib['name'] for n in tree.iter('var_array')))

def metadata(registry):
    global root,pins
    root=Path(registry).parent;pins={}
    full=extract(True);dry=extract(False)
    data={'fields':full[0],'groups':full[1],'streams':full[2],
            'dry_fields':list(dry[0]),'dry_groups':dry[1],'dry_streams':dry[2],
            'arrays':full[3],'dry_arrays':dry[3],'source_sha256':pins}
    return json.loads(json.dumps(data))
