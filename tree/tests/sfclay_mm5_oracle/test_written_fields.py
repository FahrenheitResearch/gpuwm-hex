"""Failure witnesses for the published foundation omission policy."""
from written_fields import assess

def evidence():
    return ([dict(stream='surface', field='t2m', step=3, status='compared',
                  compared_words=4, differing_words=0),
             dict(stream='surface', field='pending', step=3, status='field missing in hex',
                  compared_words=4, differing_words=4)],
            {'surface|pending|3': dict(status='field missing in hex', words=4)},
            dict(words=4, differing_words=0, files=[dict(file='surface.3.nc',
                 errors=['variable set differs: pending'])]))


def test_recorded_inventory_is_a_written_field_claim():
    result = assess(*evidence())
    assert result['passed']
    assert result['acceptance_tier'] == 'BITWISE-WRITTEN-FIELDS'
    assert not result['all_native_fields_bitwise']
    assert result['recorded_missing_samples'] == 1


def test_a_single_written_word_difference_fails():
    rows, baseline, stored = evidence()
    rows[0]['differing_words'] = 1
    assert not assess(rows, baseline, stored)['passed']


def test_a_previously_written_field_cannot_disappear():
    rows, baseline, stored = evidence()
    rows[0].update(status='field missing in hex', differing_words=4)
    assert not assess(rows, baseline, stored)['passed']


def test_new_missing_shape_and_time_fail():
    for key, value in [('compared_words', 5), ('step', 4)]:
        rows, baseline, stored = evidence()
        rows[1][key] = value
        assert not assess(rows, baseline, stored)['passed']


def test_transport_inventory_and_stored_integer_errors_fail():
    rows, baseline, stored = evidence()
    rows.append(dict(stream='unlisted', field='chunk', step=0,
                     status='extra output stream in hex', compared_words=0, differing_words=1))
    assert not assess(rows, baseline, stored)['passed']
    rows, baseline, stored = evidence()
    stored['differing_words'] = 1
    assert not assess(rows, baseline, stored)['passed']


def test_duplicate_sample_and_unknown_stored_variable_fail():
    rows, baseline, stored = evidence()
    rows.append(rows[0].copy())
    assert not assess(rows, baseline, stored)['passed']
    rows, baseline, stored = evidence()
    stored['files'][0]['errors'].append('variable set differs: t2m')
    assert not assess(rows, baseline, stored)['passed']

