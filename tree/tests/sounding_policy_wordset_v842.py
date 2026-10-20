"""Seal exact instrumented partial-list coordinate word set for the ratchet."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('snan',type=Path);p.add_argument('real',type=Path);p.add_argument('receipt',type=Path);a=p.parse_args()
w=json.loads(a.snan.read_text());real=json.loads(a.real.read_text());sets=[]
for row in w['rows']:
 if 'coordinate_words' not in row:continue
 bad=[i for i,v in enumerate(row['coordinate_words']) if (v&0x7f800000)==0x7f800000 and (v&0x7fffff)!=0]
 if bad:sets.append(dict(fixture=row['fixture'],repeat=row['repeat'],coordinate_word_indices=bad,nSoundings=row['nSoundings']))
r=dict(label='DECLARED NATIVE UNDEFINED',source='mpas_soundings.F:174-191',debug_word_sets=sets,
 real_witness_runs=[r for r in real['rows'] if r['file'].startswith('equator.')],
 exact_measured_layout=dict(nSoundings=3,nVertLevels=55,station_index_zero_based=0,
  coordinate_word_indices=[0,3],station_cells_word_indices=[0],station_owned_word_indices=[0],
  profile_shape_fortran=[6,55,3],profile_word_indices=list(range(330)),
  formatted_header_line_zero_based=3,formatted_lat_byte_columns=[8,14],formatted_lon_byte_columns=[27,34],
  formatted_profile_line_indices=list(range(6,61)),formatted_profile_byte_columns=[0,56],
  changing_file_existence_runs=['n1-r4'],unassigned_name_exclusions=0),
 interpretation='Null READ destination retains its own station element, never the preceding station. Initialize the unassigned element to zero; preserve all assigned elements.',
 conditional_excluded_set=dict(condition='location record leaves one or both numeric coordinate destinations unassigned',
 coordinates='only those individual unassigned destination words',
 dependent_words='nearest-cell index, ownership flag, and the 6*nVertLevels numerical profile words of that station only',
 formatted_bytes='the SLAT/SLON Fw.2 spans of unassigned coordinates and all six profile spans of that station; station filename/existence may vary when ownership varies',
 valid_record_exclusions=0),whole_forecast_proof=False)
a.receipt.write_text(json.dumps(r,indent=2)+'\n')
