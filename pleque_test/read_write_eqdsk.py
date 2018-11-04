if __name__ == '__main__':
    from pleque.io import _geqdsk, _readgeqdsk
    from pleque_test.testing_utils import get_test_equilibria_filenames
    from collections import *

    eqdsk_file = get_test_equilibria_filenames()[0]

    # Read Matheeesek:
    eq_1 = _readgeqdsk._readeqdsk(eqdsk_file)
    # Read Ben:
    with open(eqdsk_file, 'r') as f:
        eq_2 = _geqdsk.read(f)

    print(eq_1.keys())
    print(eq_2.keys())


    def compare(eq_1, eq_2):
        '''
        Compare content of eq_1 according to content of eq_2
        :param eq_1:
        :param eq_2:
        :return:
        '''
        for k, v in eq_1.items():
            if k in eq_2:
                if isinstance(v, Iterable):
                    if len(v) == len(eq_2[k]):
                        print("{} OK".format(k))
                    else:
                        print("{} in both, but {} != {}".format(k, len(v), len(eq_2[k])))
                else:
                    if v - eq_1[k] < 1e-3:
                        print("{} OK".format(k))
                    else:
                        print("{} in both, but {} != {}".format(k, v, eq_2[k]))
            else:
                print('! {} NOT in second'.format(k))


    print()
    print('--- Compare ben to matheesek: ---')
    compare(eq_2, eq_1)

    print()
    print('--- Compare matheesek to ben: ---')
    compare(eq_1, eq_2)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.contour(eq_1['psi'].T)

    with open('/compass/home/kripner/no-backup/python/equilibria/g_file', 'w') as f:
        _geqdsk.write(eq_2, f, 'compu_t')
