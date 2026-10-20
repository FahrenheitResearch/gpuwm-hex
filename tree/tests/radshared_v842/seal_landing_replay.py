"""Fail closed unless all stated real native-tap outputs compare bitwise.

MPAS mpas_atmphys_driver_cloudiness.F:92-370;
mpas_atmphys_o3climatology.F:59-327; LW driver:414-506,688-712,845-863.
Metadata validation only. Does not promote legacy whole-forecast receipts.
"""
import argparse
import json
from pathlib import Path

from prepare_tapped_support import sha


def qualify(directory, module_sha256):
    cloud = json.loads((directory / 'cloud-callees/matrix.json').read_text())
    driver = json.loads((directory / 'cloud-driver.json').read_text())
    support = json.loads((directory / 'tapped-support.json').read_text())
    synthetic = json.loads((directory / 'synthetic-subnormal.json').read_text())
    if cloud['module_sha256'] != module_sha256 or driver['source_sha256'] != module_sha256:
        raise ValueError('replay module source differs from requested checkout')
    for result, key in ((support, 'artifacts'), (synthetic, 'artifact_sha256')):
        hashes = [value for path, value in result[key].items() if Path(path).name == 'radshared_v842.py']
        if hashes != [module_sha256]:
            raise ValueError('support/subnormal replay module differs from requested checkout')
    if set(cloud['interfaces']) != {'cal_cldfra3', 'calc_cldfraction', 'calc_cldincidence'}:
        raise ValueError('one of the three native cloud schemes is absent')
    if not cloud['all_selected_passed'] or cloud['differing_words'] != 0 or cloud['calls'] != 11:
        raise ValueError('complete stated cloud call set does not match')
    if len(driver['captures']) != 2 or len(support['captures']) != 2:
        raise ValueError('stated native driver/support cases are incomplete')
    for result in (driver, support):
        if not result['all_outputs_bitwise_equal'] or result['differing_words'] != 0:
            raise ValueError('real native output word differs')
    expected = {'ozone_initialization', 'pressure_table', 'ozone_CAM_time',
                'ozone_monthly_date', 'ozone_vertical', 'radiation_controls'}
    expected.update('radius_' + d + '_' + p + n for d, p in (('from', 're_'), ('to', 'rre_'))
                    for n in ('cloud', 'ice', 'snow'))
    for capture in support['captures']:
        if set(capture['arrays']) != expected:
            raise ValueError('support module output set differs')
        for array in capture['arrays'].values():
            if not array['comparison_complete'] or not array['bitwise_equal'] or array['different_words'] != 0:
                raise ValueError('support output comparison incomplete or different')
    if synthetic['differing_words'] != 0 or not synthetic['subnormal_negative_control']['expected_failure_detected']:
        raise ValueError('full-module/subnormal control fails')
    return dict(real_words=cloud['total_words'] + driver['words'] + support['words'],
                differing_words=0, native_process_count=1, replay_process_count=1,
                stated_real_cloud_calls=cloud['calls'], stated_real_driver_cases=2,
                stated_real_support_cases=2, synthetic_subnormal_differing_words=0,
                complete_hidden_state_claimed=False, all_forecast_calls_claimed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    args = parser.parse_args()
    module = args.checkout / 'tree/src/hexcore/mpasphys/radshared_v842.py'
    result = qualify(args.receipts, sha(module))
    paths = (args.receipts / p for p in ('cloud-callees/matrix.json', 'cloud-driver.json',
             'tapped-support.json', 'synthetic-subnormal.json', 'oracle-source-pins.json',
             'oracle-build.log', 'transport-tests.log', 'checkout.txt'))
    result.update(schema='p21-revised-landing-module-replay/v1', passed=True,
                  artifacts={str(p): sha(p) for p in paths}, module_sha256=sha(module))
    (args.receipts / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
