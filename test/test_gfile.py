
def compare_gfile(gfile_name):
    from pleque.io._geqdsk import read_as_equilibrium, read, data_as_ds

    with open(gfile_name, 'r') as f:
        data = read(f)
        eq_ds = data_as_ds(data)
    with open(gfile_name, 'r') as f:
        eq = read_as_equilibrium(f)

    print(eq_ds.keys())

def test_gfile():
    from .testing_utils import get_test_equilibria_filenames, get_test_cases_number

    g_files = get_test_equilibria_filenames()
    n_files = get_test_cases_number()

    compare_gfile(g_files[5])


    pass