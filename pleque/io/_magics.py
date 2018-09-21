def read_write_geqdsk(file_in, file_out):
    '''
    Read the equilibrium file and then save it again. In principle,
    newly written file may be according to g-file standard.
    :param file_in:
    :param file_out:
    :return:
    '''
    from . import _geqdsk

    with open(file_in, 'r') as f:
        eq_in = _geqdsk.read(f)

    with open(file_out, 'w') as f:
        _geqdsk.write(eq_in, f)
