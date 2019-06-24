
def compare_gfile(gfile_name):
    from pleque.io._geqdsk import read_as_equilibrium, read, data_as_ds
    import numpy as np

    with open(gfile_name, 'r') as f:
        data = read(f)
        eq_ds = data_as_ds(data)
    with open(gfile_name, 'r') as f:
        eq = read_as_equilibrium(f)

    print(eq_ds.keys())

    # import matplotlib.pyplot as plt
    #
    # eq._plot_overview()
    #
    # plt.figure()
    # psi_n = np.linspace(0, 1, 20, endpoint=False)
    # plt.plot(psi_n, eq.q(psi_n=psi_n))
    #
    # plt.show()

def test_gfile():
    from pleque.tests.utils import get_test_equilibria_filenames, get_test_cases_number

    g_files = get_test_equilibria_filenames()
    n_files = get_test_cases_number()

    compare_gfile(g_files[4])


def test_from_to_gfile(equilibrium):
    file_name = '/tmp/g{:d}.{:.0f} {}'.format(equilibrium.shot, equilibrium.time, equilibrium.time_unit)

    equilibrium.to_geqdsk(file_name)

    eq2 = read(file_name)
    os.remove(file_name)

    assert np.isclose(equilibrium.magnetic_axis.R, eq2.magnetic_axis.R, atol=1e-5, rtol=1e-4)
    assert np.isclose(equilibrium.magnetic_axis.Z, eq2.magnetic_axis.Z, atol=1e-5, rtol=1e-4)

    pass