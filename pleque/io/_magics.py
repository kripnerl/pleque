from pleque.io import _geqdsk, readers
import matplotlib.pyplot as plt


def read_write_geqdsk(file_in, file_out):
    '''
    Read the equilibrium file and then save it again. In principle,
    newly written file may be according to g-file standard.
    :param file_in:
    :param file_out:
    :return:
    '''

    with open(file_in, 'r') as f:
        eq_in = _geqdsk.read(f)

    with open(file_out, 'w') as f:
        _geqdsk.write(eq_in, f)

    # with open(file_out, 'r') as f:
    #     eq_out = _geqdsk.read(f)
    #
    # fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    # ax = axs[0]
    # ax.pcolormesh(eq_in['psi'].T)
    # ax = axs[1]
    # ax.pcolormesh(eq_out['psi'].T)
    #
    # plt.show()

def read_write_geqsk_flip_direction(file_in, file_out, flip=True, plot=False):
    """
    Read possibly corrupted g-eqdsk file and change direction of B and I_plasma.

    :param file_in: file name
    :param file_out: file name
    :param flip: bool, if true flip direction of I_plasma and B_tor.
    :param plot: bool, if true plot
    :return:
    """

    with open(file_in, 'r') as f:
        eq_in = _geqdsk.read(f)

    if flip:
        # Change of current -> change of psi
        eq_in['psi'] = - eq_in['psi']
        eq_in['simagx'] = - eq_in['simagx']
        eq_in['sibdry'] = - eq_in['sibdry']

        eq_in['pprime'] = - eq_in['pprime']
        eq_in['FFprime'] = - eq_in['FFprime']

        # Change of toroidal magnetic field
        eq_in['F'] = - eq_in['F']

    with open(file_out, 'w') as f:
        _geqdsk.write(eq_in, f)

    if plot:

        eq = readers.read_geqdsk(file_out)

        eq.plot_geometry()
        plt.show()

