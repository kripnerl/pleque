
def test_equilibria():
    from test.testing_utils import load_testing_equilibrium, get_test_cases_number
    import matplotlib.pyplot as plt

    N_cases = get_test_cases_number()

    for i in range(N_cases):
        eq = load_testing_equilibrium(i)
        eq._



    pass