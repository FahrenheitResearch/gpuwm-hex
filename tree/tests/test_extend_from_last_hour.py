"""Extend a run: start from the source's LAST forecast hour and run on, the free run named in every receipt.

The breakage this covers: a run past a source's horizon had no front door ("start a hex 15 km global at IFS HRES
f360 and run another 120 h"), and nothing said which hours the source covered and which were our own free run.
"""
import datetime as dt

import pytest

from hexcore import global_door
from hexcore.cli import build_parser
from hexcore.init_sources import load_table, resolve_source
from hexcore.source_horizon import FREE_RUN_NOTE, SourceHorizonError, beyond_source, last_posted_lead


@pytest.mark.parametrize('source,hour,last', [('ifs', 12, 360), ('ifs', 18, 144), ('ifs-0p1', 0, 360),
                                              ('aifs', 6, 360), ('gfs', 18, 384)])
def test_last_posted_lead_comes_from_the_table(source, hour, last):
    assert last_posted_lead(resolve_source(load_table(), source), hour) == last


def test_an_analysis_source_has_no_last_hour():
    with pytest.raises(SourceHorizonError, match='analysis source'):
        last_posted_lead(resolve_source(load_table(), 'era5'), 0)


def test_beyond_source_names_the_free_run():
    row = resolve_source(load_table(), 'ifs')
    cycle = dt.datetime(2026, 10, 23, 12)
    assert beyond_source(row, cycle, 240, 120) is None
    extension = beyond_source(row, cycle, 360, 120)
    assert extension['free_running_hours'] == 120 and extension['source_last_lead_hours'] == 360
    assert extension['free_running_from_utc'] == '2026-11-07T12:00:00Z' and extension['note'] == FREE_RUN_NOTE


def test_a_globe_from_the_last_ifs_hour_runs_120_h_free(tmp_path, monkeypatch):
    monkeypatch.setattr(global_door, '_visible_cards', lambda: [])
    args = build_parser().parse_args([
        'global', '--row', 'global-60km', '--source', 'ifs', '--cycle', '2026-10-23T12', '--from-last-hour',
        '--hours', '120', '--work-dir', str(tmp_path / 'run'), '--cache-dir', str(tmp_path / 'cache'),
        '--stage', 'plan'])
    value = global_door.plan(args)
    assert value['lead_hours'] == 360 and value['start_time'] == '2026-11-07_12:00:00'
    assert value['beyond_source']['free_running_hours'] == 120
    label = value['commands']['forecast'][value['commands']['forecast'].index('--init-source') + 1]
    assert 'ifs 2026-10-23T12 f360' in label and 'free-running beyond the source' in label
    # every sounding names the free run, top right
    assert args.brand == 'ifs 2026-10-23T12 f360 · free run past f360'


def test_from_last_hour_and_a_lead_are_one_choice(tmp_path):
    args = build_parser().parse_args([
        'global', '--row', 'global-60km', '--source', 'ifs', '--cycle', '2026-10-23T12', '--from-last-hour',
        '--lead-hours', '240', '--work-dir', str(tmp_path / 'run'), '--stage', 'plan'])
    with pytest.raises(Exception, match='two different starts'):
        global_door.plan(args)
