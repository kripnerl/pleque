import numpy as np

def test_equilibria():
    from test.testing_utils import load_testing_equilibrium, get_test_cases_number
    from pleque.utils.plotting import plot_extremes
    import matplotlib.pyplot as plt
    import numpy as np

    N_cases = get_test_cases_number()

    o_points = [[0.909, 0.0144], [0.908,  0.0675], [9.103e-01, 2.647e-07]]
    x_points = [[0.750, -0.498], [0.750, -0.449], [0.738, 0.499]]

    for i in range(N_cases):

        eq = load_testing_equilibrium(i)
        #eq._plot_overview()
        #plot_extremes(eq)

        assert np.isclose(x_points[i], eq._x_point, rtol=1e-3, atol=1e-3)
        assert np.isclose(o_points[i], eq._mg_axis, rtol=1e-3, atol=1e-3)

        print('mg axis = {}'.format(eq._mg_axis))
        print('x point = {}'.format(eq._x_point))



    plt.show()
