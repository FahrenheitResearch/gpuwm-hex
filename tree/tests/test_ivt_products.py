"""Product admission for transport, low-level winds and the jet sequence."""
from hexcore import products_door


def test_column_transport_and_jet_maps_are_default_products():
    # "ivt_column" is the renderer's model-column recipe; plain "ivt" is its 1000-300 hPa one, which a hex
    # history cannot feed.
    assert "ivt" not in products_door.GLOBAL_PRODUCTS
    assert {"ivt_column", "ivt_vectors", "isotherm_height_0c", "250mb_height_winds", "300mb_height_winds"} <= set(products_door.GLOBAL_PRODUCTS)
    assert {"925mb_height_winds", "850mb_height_winds"} <= set(products_door.REGIONAL_PRODUCTS)
    assert len(products_door.GLOBAL_PRODUCTS) == len(set(products_door.GLOBAL_PRODUCTS))
    assert len(products_door.REGIONAL_PRODUCTS) == len(set(products_door.REGIONAL_PRODUCTS))
