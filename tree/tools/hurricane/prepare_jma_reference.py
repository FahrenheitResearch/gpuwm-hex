"""Stage JMA best-track positions/pressure without mixing wind averaging periods.

Source format: https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/Besttracks/e_format_bst.html
Terms: https://www.jma.go.jp/jma/en/copyright.html (PDL 1.0 / CC BY 4.0).
This creates a derivative ATCF file for the existing shared scorer. Ten-minute
wind remains in the audit JSON and is deliberately absent from the ATCF output.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import zipfile


def convert(archive, jma_id, storm, out):
    if not re.fullmatch(r'\d{4}', jma_id) or not re.fullmatch(r'WP\d{6}', storm):
        raise ValueError('require a four-digit JMA ID and a WPnnYYYY identifier')
    season = int(storm[-4:])
    if int(jma_id[:2]) != season % 100:
        raise ValueError('JMA and ATCF seasons differ')
    raw = Path(archive).read_bytes()
    with zipfile.ZipFile(archive) as z:
        text = z.read('bst_all.txt').decode('ascii')
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('66666 '+jma_id+' ')]
    if len(starts) != 1:
        raise ValueError('expected exactly one matching storm header')
    begin = starts[0]
    header = lines[begin]
    count = int(header[12:15])
    selected = lines[begin+1:begin+1+count]
    if len(selected) != count or any(line.startswith('66666') for line in selected):
        raise ValueError('truncated or inconsistent JMA record count')
    if begin+1+count < len(lines) and not lines[begin+1+count].startswith('66666'):
        raise ValueError('JMA record count does not reach the next storm header')
    rows, atcf = [], []
    for line in selected:
        if line[9:12] != '002':
            raise ValueError('unexpected JMA analysis row indicator')
        year = next((y for y in (season-1, season, season+1) if y % 100 == int(line[:2])), None)
        if year is None:
            raise ValueError('analysis year differs from the identified season')
        valid = f'{year:04d}'+line[2:8]
        datetime.strptime(valid, '%Y%m%d%H')
        grade = int(line[13:14])
        lat, lon, pressure = int(line[15:18]), int(line[19:23]), int(line[24:28])
        wind = int(line[33:36])
        if not (0 <= lat <= 900 and 0 <= lon <= 1800 and 800 <= pressure <= 1100):
            raise ValueError('invalid JMA location or pressure')
        status = {2:'TD', 3:'TS', 4:'TS', 5:'TY', 6:'EX'}.get(grade, 'XX')
        rows.append(dict(valid=valid, lat=lat/10, lon=lon/10, pressure_hpa=pressure,
                         wind_10minute_kt=wind or None, grade=grade, status=status))
        atcf.append(f'WP, {storm[2:4]}, {valid}, , BEST, 0, {lat:03d}N, {lon:04d}E, , {pressure}, {status},')
    if len({r['valid'] for r in rows}) != len(rows):
        raise ValueError('duplicate JMA times')
    out = Path(out);out.mkdir(parents=True, exist_ok=True)
    data = ('\n'.join(atcf)+'\n').encode()
    (out/'track-pressure.dat').write_bytes(data)
    (out/'source-record.txt').write_text('\n'.join([header,*selected])+'\n')
    receipt = dict(source_archive=str(Path(archive).resolve()), archive_sha256=hashlib.sha256(raw).hexdigest(),
        source_url='https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/Besttracks/bst_all.zip',
        attribution='Derived from Japan Meteorological Agency best-track data; ATCF conversion and wind exclusion by this tool.',
        terms_url='https://www.jma.go.jp/jma/en/copyright.html', license='PDL 1.0, usable under CC BY 4.0',
        jma_id=jma_id, storm=storm, source_header=header, rows=rows,
        wind_scoring='Excluded: JMA ten-minute wind must not be scored as one-minute wind',
        output_sha256=hashlib.sha256(data).hexdigest())
    (out/'source.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--jma-id',required=True)
    p.add_argument('--storm',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=convert(a.archive,a.jma_id,a.storm,a.out)
    print(json.dumps(dict(rows=len(r['rows']),output_sha256=r['output_sha256'])))


if __name__=='__main__':main()
