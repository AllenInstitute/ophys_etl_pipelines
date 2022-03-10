import pytest
from ophys_etl.modules.mesoscope_splitting.mixins import (
    IntToZMapperMixin)


@pytest.mark.parametrize("atol", [0.01, 0.001, 0.0001])
def test_int_to_z_mapper(atol):
    mapper = IntToZMapperMixin()
    f0 = 4.1
    f1 = f0 + 1.01*atol
    f2 = f0 + 0.5*atol
    f3 = 5.7
    f4 = f0 + 0.6*atol

    i0 = mapper._int_from_z(z_value=f0, atol=atol)
    i1 = mapper._int_from_z(z_value=f1, atol=atol)
    i2 = mapper._int_from_z(z_value=f2, atol=atol)
    i3 = mapper._int_from_z(z_value=f3, atol=atol)
    i4 = mapper._int_from_z(z_value=f4, atol=atol)

    assert i0 != i1
    assert i0 != i3
    assert i0 == i2
    assert i1 != i3
    assert i4 == i1

    assert len(mapper._int_from_z_lookup) == 3
    assert len(mapper._z_from_int_lookup) == 3

    for pair in ((i0, f0), (i1, f1), (i3, f3)):
        actual = mapper._z_from_int(ii=pair[0])
        assert abs(actual-pair[1]) < 1.0e-10
