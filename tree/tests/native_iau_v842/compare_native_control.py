"""Rust word-digest comparison of regenerated controlled native forecasts."""
from pathlib import Path
import hashlib,json,os,subprocess
root=Path(__file__).resolve().parents[4]
old=json.loads((root/'receipts/iau-whole-a449273-native-clocks.json').read_text())
run=root/'n1out/iau-active-native-landed-compare'
output=root/'n1out/iau-landed-native-repeat-words.json'
actual={p.name for p in run.glob('*.nc') if not p.is_symlink()}
if actual!=set(old['files']):raise SystemExit('native repeat output inventory differs')
rows=[]; different=[]; words=0
for name,expected in sorted(old['files'].items()):
 result=subprocess.run([str(root/'scratch/iau-native-runtime/mpas-native-word-compare'),'--field-digests',str(run/name)],capture_output=True,text=True,check=True)
 inventory=json.loads(result.stdout)
 before={r['field']:r for r in expected['inventory']['variables']}
 after={r['field']:r for r in inventory['variables']}
 if set(before)!=set(after) or inventory['xtime']!=expected['inventory']['xtime']:
  raise SystemExit('native repeat field inventory or clocks differ: '+name)
 for field,row in sorted(after.items()):
  previous=before[field]
  same=all(row[k]==previous[k] for k in ('netcdf_type','shape','dimensions','words','word_sha256'))
  record=dict(file=name,field=field,words=row['words'],identical=same,
              prior_word_sha256=previous['word_sha256'],word_sha256=row['word_sha256'])
  rows.append(record);words+=row['words']
  if not same:different.append(record)
receipt=dict(schema='mpas.iau-controlled-native-repeat/v1',host=os.uname().nodename,
  process_count=1,openmp_threads=1,duration_seconds=86400,files=len(actual),field_records=len(rows),
  raw_stored_words=words,all_field_words_identical=not different,
  different_words=0 if not different else None,mismatched_field_records=different,
  scope='controlled IAU fixture, not analysis-minus-background acceptance or Hex identity',
  prior_host_b_clocks_sha256=hashlib.sha256((root/'receipts/iau-whole-a449273-native-clocks.json').read_bytes()).hexdigest(),fields=rows)
output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('fields','mismatched_field_records')}))
raise SystemExit(bool(different))
