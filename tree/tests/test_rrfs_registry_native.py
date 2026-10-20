"""Audit all dycore defaults against the complete pinned fork Registry."""
from hexcore.config_rrfs import RRFSDryDycoreConfig

def test_complete_fork_registry_dynamics_defaults():
    """Authenticate every registered dynamics default, not just epssm.

    MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:73-321 at 1d6e370e.
    """
    import os
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from rrfs_native.native_source import read_pinned_source
    import site_support
    root=site_support.rrfs_source('RRFS_MPAS_SOURCE')
    registry=ET.fromstring(read_pinned_source(root,root/'src/core_atmosphere/Registry.xml'))
    config=RRFSDryDycoreConfig()
    compared=[]
    for record in registry.iter('nml_record'):
        if record.get('name') not in ('nhyd_model','damping','limited_area'):
            continue
        for option in record.findall('nml_option'):
            name=option.get('name')
            if not hasattr(config,name):
                continue
            text=option.get('default_value'); kind=option.get('type')
            expected=(text.lower().strip('.')=='true' if kind=='logical' else
                      int(text) if kind=='integer' else float(text) if kind=='real' else text)
            assert getattr(config,name)==expected, name
            compared.append(name)
    assert {'config_epssm','config_dt','config_time_integration_order',
            'config_positive_definite','config_horiz_mixing','config_lbc_w'} <= set(compared)
    config.validate()
