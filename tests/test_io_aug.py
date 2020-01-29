import pytest
map_equ = pytest.importorskip('map_equ')


def test_read_aug_eq():
    from pleque.io.aug import read_aug_eq
    from pleque.io.tools import EquilibriaTimeSlices

    eqs = read_aug_eq(35802, 'EQI')
    assert isinstance(eqs, EquilibriaTimeSlices)
    eq = eqs.get_time_slice(2)   # [s]
    print(eq)
