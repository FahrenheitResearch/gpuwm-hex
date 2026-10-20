"""Compare full pristine MPAS/ESMF recurring alarms with the Rust bridge."""
from pathlib import Path
import argparse,json,subprocess
from decimal import Decimal

def instant(text):
    parts=text.strip().split('.')
    return parts[0],Decimal('0.'+parts[1]) if len(parts)>1 else Decimal(0)

def replay(native,rust):
    rows=[]
    for start,interval,now in [
        ('2000-01-01_00:00:00','00:30:00','2000-01-01_00:00:00'),
        ('2000-01-01_00:00:00','00:30:00','2000-01-01_00:29:59'),
        ('2000-01-01_00:00:00','00:30:00','2000-01-01_00:30:00'),
        ('2000-01-01_00:00:00','00:30:00','2000-01-01_11:14:13'),
        ('2000-02-28_23:59:59','2','2000-02-29_00:00:03'),
        ('1900-02-28_23:59:59','1:00','1900-03-01_00:05:07'),
        ('2000-12-31_23:30:00','1_00:00:00','2001-01-05_23:30:00'),
        ('2000-01-31_00:00:00','1-0_00:00:00','2000-03-31_00:00:00'),
        ('2000-02-29_00:00:00','1-0-0_00:00:00','2004-02-29_00:00:00'),
        ('2000-01-01_00:00:00','00:00:00','2000-01-02_00:00:00'),
        ('2000-01-01_00:00:00.25','00:00:02.5','2000-01-01_00:00:10.25'),
        ('2000-01-01_00:00:00','0.123456789','2000-01-01_00:00:01.1'),
        ('2000-02-29_12:30:45.75','0-1-1_01:00:00.25','2000-06-07_12:30:45'),
    ]:
        before=subprocess.run([str(native),start,interval,start,'keep'],capture_output=True,text=True,check=True).stdout.splitlines()
        previous=before[-2]
        for reset in (False,True):
            a=subprocess.run([str(native),start,interval,now,'reset' if reset else 'keep'],capture_output=True,text=True,check=True).stdout.splitlines()
            request=dict(now=now,interval=interval,previous=previous,reset=reset)
            b=json.loads(subprocess.run([str(rust),'clock'],input=json.dumps(request),capture_output=True,text=True,check=True).stdout)
            expected=dict(ringing=a[-3]=='true',previous=a[-2],next=a[-1])
            assert expected['ringing']==b['ringing'] and instant(expected['previous'])==instant(b['previous']) and instant(expected['next'])==instant(b['next']),(request,expected,b)
            rows.append(dict(start=start,request=request,native=expected,rust=b,matching=True))
    return dict(tier='strict copied pristine MPAS/ESMF versus Rust clock',whole_forecast_proof=False,rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('native',type=Path);p.add_argument('rust',type=Path);p.add_argument('receipt',type=Path);a=p.parse_args()
    r=replay(a.native.resolve(),a.rust.resolve());a.receipt.write_text(json.dumps(r,indent=2)+'\n');print(f"{len(r['rows'])} native clock checks matched")
