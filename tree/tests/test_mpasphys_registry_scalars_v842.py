"""Registry scalar selection, including physics without active microphysics.

MPAS source: Registry.xml:1701-1754; mpas_atmphys_interface.F:284-305.
"""
import pytest
from hexcore.errors import ConfigurationRefusal
from hexcore.cuda_physics_prep_v841 import _validate_scalar_names


@pytest.mark.parametrize('names', [('qv',), ('qv','qc','qr'),
                                  ('qv','qc','qr','qi','qs','qg'),
                                  ('qv','qc','qr','qi','qs','qg','ni','nr','nc','nifa','nwfa')])
def test_active_registry_inventory_is_a_complete_preparation_declaration(names):
    assert _validate_scalar_names(names, len(names), registry_scalar_names=names) == names


@pytest.mark.parametrize('names,declared', [
    (('qv',),('qv','qc','qr')),
    (('qc','qv'),('qc','qv')),
    (('qv','qv'),('qv','qv')),
    ((),()),
])
def test_registry_preparation_rejects_mismatched_extent_order_and_aliases(names, declared):
    with pytest.raises(ValueError):
        _validate_scalar_names(names,len(names),registry_scalar_names=declared)


def test_existing_arwen_preparation_still_requires_its_species_table():
    with pytest.raises(ConfigurationRefusal):
        _validate_scalar_names(('qv',),1)


def test_registry_extent_cannot_drop_a_resident_scalar():
    with pytest.raises(ValueError):
        _validate_scalar_names(('qv',),2,registry_scalar_names=('qv',))
