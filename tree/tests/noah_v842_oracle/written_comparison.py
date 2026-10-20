"""Check written-word coverage without a forecast runner or host dependencies."""

def written_failures(fields, baseline, stored, subset=False):
    expected={(f['stream'],f['field'],f['step']):f for f in baseline}
    seen={};failures=[]
    for row in fields:
        key=(row['stream'],row['field'],row['step'])
        if key in seen:failures.append('duplicate sample '+str(key))
        seen[key]=row;old=expected.get(key)
        if old is None:failures.append('sample absent from accepted foundation '+str(key));continue
        if row['status']=='compared':
            if row['differing_words']:failures.append('computed words differ '+str(key))
        elif row['status']!='field missing in hex' or old['status']!='field missing in hex' or row['compared_words']!=old['compared_words']:
            failures.append('new or grown omission '+str(key))
        if old['status']=='compared' and (row['status']!='compared' or row['compared_words']!=old['compared_words']):
            failures.append('previously written native coverage changed '+str(key))
    if not subset and seen.keys()!=expected.keys():failures.append('complete native sample inventory changed')
    if not fields or not any(f['status']=='compared' for f in fields):failures.append('no actual written comparison')
    if not stored.get('files') or stored.get('differing_words')!=0:failures.append('stored comparison absent or nonzero')
    for file in stored.get('files',[]):
        permitted={key[1] for key,row in expected.items() if key[0]==file['file'].split('.')[0] and row['status']=='field missing in hex'}
        if file.get('error'):failures.append(file['file']+': '+file['error'])
        for error in file.get('errors',[]):
            if not error.startswith('variable set differs: ') or error.removeprefix('variable set differs: ') not in permitted:
                failures.append(file['file']+': '+error)
    return failures
