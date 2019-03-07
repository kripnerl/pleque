from pleque.io import _geqdsk
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
