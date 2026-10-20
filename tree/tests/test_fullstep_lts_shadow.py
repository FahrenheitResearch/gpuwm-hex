"""CPU proofs of the shadow instrument and clock, not active forecast gates."""
from fractions import Fraction
from types import SimpleNamespace
import numpy as np
import pytest
from hexcore.config_fullstep_lts import FullStepLtsConfig, validate_fullstep_lts_request
from hexcore.courant_fullstep_lts import column_transport_rates
from hexcore.errors import ConfigurationRefusal
from hexcore.fullstep_lts import build_macro_plan, clock_ticks, buffer_rates, transport_ceiling, classify_fullstep_rates


def fixture():
    return dict(rho_old=np.ones((2, 3)), uh_avg=np.array([[.2, -.3, .4], [.1, -.2, .3]]),
                ww_avg=np.array([[0., 0., 0.], [.5, -.1, 0.], [0., 0., 0.]]),
                rdzw=np.array([1., 2.]), edges_on_cell=np.array([[0,2], [0,1], [1,2]]),
                n_edges_on_cell=np.full(3,2), acoustic_sign=np.array([[1.,-1.], [-1.,1.], [-1.,1.]]),
                dv_edge=np.array([2.,3.,4.]), area_cell=np.array([2.,3.,4.]))


def test_donor_is_independent_incidence_ledger():
    args = fixture()
    actual = column_transport_rates(**args)
    mass = args['rho_old']*args['area_cell'][None,:]/args['rdzw'][:,None]
    outgoing = np.zeros_like(mass)
    for edge, (a,b) in enumerate([(0,1),(1,2),(2,0)]):
        for k in range(2):
            packet = args['uh_avg'][k,edge]*args['dv_edge'][edge]/args['rdzw'][k]
            outgoing[k, a if packet >= 0 else b] += abs(packet)
    for c in range(3):
        packet = args['ww_avg'][1,c]*args['area_cell'][c]
        outgoing[0 if packet >= 0 else 1,c] += abs(packet)
    expected = outgoing/mass
    np.testing.assert_allclose(actual.combined, expected.max(axis=0), atol=0, rtol=1e-15)
    np.testing.assert_array_equal(actual.level, expected.argmax(axis=0))


def test_combined_max_is_not_sum_of_separate_maxima():
    args = fixture()
    args['uh_avg'][:] = 0
    args['uh_avg'][0,0] = .6
    args['ww_avg'][1,0] = -.4
    result = column_transport_rates(**args)
    assert result.combined[0] == .8
    assert result.horizontal[0]+result.vertical[0] == 1.4


@pytest.mark.parametrize('name', ['rho_old','uh_avg','ww_avg','rdzw','dv_edge','area_cell'])
def test_nonfinite_never_dropped(name):
    args = fixture()
    args[name].flat[0] = np.nan
    with pytest.raises(ValueError, match=name):
        column_transport_rates(**args)


def test_padding_ignored_and_active_bad_edge_rejected():
    args = fixture()
    args['edges_on_cell'] = np.column_stack((args['edges_on_cell'], [-99]*3))
    args['acoustic_sign'] = np.column_stack((args['acoustic_sign'], [np.nan]*3))
    column_transport_rates(**args)
    args['n_edges_on_cell'][0] = 3
    with pytest.raises(ValueError):
        column_transport_rates(**args)


def test_event_window_and_initial_warmup_tail():
    for start, expected in [(0, [288]*6+[72]), (36, [288]*6+[36])]:
        durations = []
        t = start
        while t < 1800:
            plan = build_macro_plan(t, [1,2,4,8,16], [1800], base_dt=18)
            durations.append((plan.end_tick-plan.start_tick)//48)
            for rate in plan.effective_rates:
                spans = sorted((a,b) for r,a,b in plan.intervals if r == rate)
                assert spans[0][0] == plan.start_tick
                assert spans[-1][1] == plan.end_tick
                assert all(b == c for (_,b),(c,_) in zip(spans, spans[1:]))
                assert all(b-a == rate*plan.base_ticks for a,b in spans)
            t = plan.end_tick//48
        assert durations == expected


def test_clock_refuses_rounded_events_and_exact_split_offsets():
    assert clock_ticks(.125, base=True) == 6
    assert clock_ticks(Fraction(1,48)) == 1
    with pytest.raises(ValueError):
        clock_ticks(.1, base=True)
    with pytest.raises(ValueError, match='shift radiation'):
        build_macro_plan(0,[1,2,4],[1801],base_dt=18)


def test_buffers_are_simultaneous_and_do_not_mutate_input():
    rates = np.array([1,16,16,16,16])
    coe = np.array([[0,1],[1,2],[2,3],[3,4]])
    assert buffer_rates(rates,coe).tolist() == [1,1,1,16,16]
    assert rates.tolist() == [1,16,16,16,16]


def test_optimistic_transport_ceiling_reports_infeasible_base():
    rates, bad = transport_ceiling([16,16,16],[.001,.02,.1],18)
    assert rates.tolist() == [16,2,1]
    assert bad.tolist() == [False,False,True]


def test_config_default_and_active_request():
    assert validate_fullstep_lts_request(SimpleNamespace()) == FullStepLtsConfig()
    # Retired refusal: active mode stopped here while physics had no
    # per-class clock.  The request now validates; the class executor owns
    # the refusals that remain.
    assert validate_fullstep_lts_request(FullStepLtsConfig(True, 'active')).fullstep_lts_mode == 'active'
    with pytest.raises(ConfigurationRefusal, match='ignored'):
        validate_fullstep_lts_request(FullStepLtsConfig(fullstep_lts_max_rate=4))
    assert validate_fullstep_lts_request(FullStepLtsConfig(True,'active',1)).fullstep_lts_max_rate == 1


def test_disabled_and_forced_one_dispatch_never_read_driver():
    from hexcore.fullstep_lts_dispatch import attach_fullstep_lts
    class Poison:
        def __getattribute__(self,name):
            raise AssertionError(f'unexpected driver read: {name}')
    assert attach_fullstep_lts(Poison(),FullStepLtsConfig()) is None
    assert attach_fullstep_lts(Poison(),FullStepLtsConfig(True,'active',1)) is None


@pytest.mark.parametrize('value', [0,3,True,32])
def test_non_dyadic_config_refused(value):
    with pytest.raises(ConfigurationRefusal):
        validate_fullstep_lts_request(FullStepLtsConfig(True, fullstep_lts_max_rate=value))


def test_incomplete_budgets_cannot_admit_active_rates():
    geom = dict(dc_edge=np.array([1.,4.,4.]), cells_on_edge=np.array([[0,1],[1,2],[2,0]]), n_cells=3)
    with pytest.raises(KeyError):
        classify_fullstep_rates(geom, {'transport':np.zeros(3)}, dt=18)


def test_demotion_last_two_samples_and_headroom_promotion():
    geom=dict(dc_edge=np.array([1.]+[16.]*8),cells_on_edge=np.array([[i,i+1] for i in range(9)]),n_cells=10)
    def budgets(transport):
        return {name:np.full(10,transport if name=='transport' else 0.) for name in
                ('transport','advection','acoustic','mixing','physics')}
    rates,history=classify_fullstep_rates(geom,budgets(.3),dt=1)
    assert rates[-1]==2
    # The previous completed sample still controls admission.
    rates,history=classify_fullstep_rates(geom,budgets(0),history,dt=1)
    assert rates[-1]==2
    rates,history=classify_fullstep_rates(geom,budgets(0),history,dt=1)
    assert rates[-1]==2
    rates,history=classify_fullstep_rates(geom,budgets(0),history,dt=1)
    assert rates[-1]==4
    rates,history=classify_fullstep_rates(geom,budgets(.5),history,dt=1)
    assert rates[-1]==1


def test_reclassification_does_not_repeatedly_erode_unchanged_buffers():
    geom=dict(dc_edge=np.array([1.]+[16.]*8),cells_on_edge=np.array([[i,i+1] for i in range(9)]),n_cells=10)
    budgets={name:np.zeros(10) for name in ('transport','advection','acoustic','mixing','physics')}
    rates,history=classify_fullstep_rates(geom,budgets,dt=1)
    first=rates.copy()
    for _ in range(10):
        rates,history=classify_fullstep_rates(geom,budgets,history,dt=1)
        np.testing.assert_array_equal(rates,first)


def test_census_acoustic_budget_is_one_sub_step_of_the_run():
    """The census holds the acoustic rate against dt/(3N): N=4 (the default) admits what N=2 demotes."""
    geom=dict(dc_edge=np.array([1.]+[16.]*8),cells_on_edge=np.array([[i,i+1] for i in range(9)]),n_cells=10)
    def budgets(acoustic):
        return {name:np.full(10,acoustic if name=='acoustic' else 0.) for name in
                ('transport','advection','acoustic','mixing','physics')}
    target=FullStepLtsConfig().fullstep_lts_target_acoustic
    # dt*rate*acoustic <= 3N*target: at 0.5 s^-1 and dt 1, N=2 allows rate 8 (4 <= 4.2), N=4 allows 16 (8 <= 8.4).
    assert 8*.5 <= 6*target < 16*.5 <= 12*target
    two,_=classify_fullstep_rates(geom,budgets(.5),dt=1,acoustic_substeps=2)
    four,_=classify_fullstep_rates(geom,budgets(.5),dt=1,acoustic_substeps=4)
    default,_=classify_fullstep_rates(geom,budgets(.5),dt=1)
    assert two.max()==8 and four.max()==16
    np.testing.assert_array_equal(default,four)
    # the base step itself is refused when one sub-step exceeds the bound, at the run's N
    with pytest.raises(ValueError, match='acoustic'):
        classify_fullstep_rates(geom,budgets(4.3),dt=1,acoustic_substeps=2)
    classify_fullstep_rates(geom,budgets(4.3),dt=1,acoustic_substeps=4)
    with pytest.raises(ValueError, match='even acoustic sub-step count'):
        classify_fullstep_rates(geom,budgets(0.),dt=1,acoustic_substeps=3)
