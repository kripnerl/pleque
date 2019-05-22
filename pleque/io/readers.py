
from ._geqdsk import read_as_equilibrium


def read_geqdsk(filename, cocos=1):
    """
    Read a G-EQDSK formatted equilibrium file

    Format is specified here:
    https://fusion.gat.com/theory/Efitgeqdsk

    cocos   -
    :param str filename:
    :param int cocos:
        COordinate COnventions. Not fully handled yet,
        only whether psi is divided by 2pi or not.
        if < 10 then psi is divided by 2pi, otherwise not.
    :return: instance of `Equilibrium`
    """

    with open(filename, 'r') as f:
        eq = read_as_equilibrium(f, cocos)

    return eq
