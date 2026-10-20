"""The MPAS suite route refuses a local-time-stepping switch by name instead of ignoring it."""
from argparse import Namespace

from hexcore.forecast_door import mpas_suite_lts_refusal


def test_mpas_suite_route_refuses_lts_flags_by_name() -> None:
    for flags, named in (({"fullstep_local_timestep": True}, "--fullstep-local-timestep"),
                         ({"local_timestep": True}, "--local-timestep")):
        message = mpas_suite_lts_refusal(Namespace(**flags), environ={})
        assert message is not None and message.startswith(named)
        assert "fixed step only" in message and "different time scheme" in message
    assert mpas_suite_lts_refusal(Namespace(fullstep_local_timestep=False, local_timestep=False), environ={}) is None
    # A moving segment carries the controller's own LTS flags; the moving-route admission judges it.
    assert mpas_suite_lts_refusal(Namespace(fullstep_local_timestep=True),
                                  environ={"GPUWM_HEX_MOVING_CONTROL": "/x/control.json"}) is None
