
from ._geqdsk import read_as_equilibrium


def read_geqdsk(filename, cocos=3, init_method='hints', **kwargs):
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
    :param init_method: str One of ("full", "hints", "fast").
                            If "full" no hints are taken and module tries to recognize all critical points itself.
                            If "hints" module use given optional arguments as a help with initialization.
                            If "fast" module use given optional arguments as final and doesn't try to correct.
    :return: instance of `Equilibrium`
    """

    with open(filename, 'r') as f:
        eq = read_as_equilibrium(f, cocos, init_method, **kwargs)

    return eq


