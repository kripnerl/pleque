import numpy as np

def test_equilibria():
    from pleque_test.testing_utils import load_testing_equilibrium, get_test_cases_number, get_test_equilibria_filenames
    from pleque.io.readers import read_geqdsk
    from numpy import array
    from pleque.utils.plotting import plot_extremes
    import matplotlib.pyplot as plt
    import numpy as np

    N_cases = get_test_cases_number()
    gfiles = get_test_equilibria_filenames()

    o_points = [array([0.90891258, 0.01440663]), array([0.9083923, 0.06746012]), array([9.10399169e-01, 2.64745319e-07]),
                array([0.56788956, 0.00523984]), array([6.36134207, 0.58880498]), array([0.55436531, 0.02267376]),
                array([0.56631465, 0.01856815])]
    x_points = [array([0.74987394, -0.49864919]), array([0.74987669, -0.44865779]), array([0.73877241, 0.49947451]), None,
                array([5.07249766, -3.44214432]), array([0.47302589, -0.33277101]), array([0.46133564, -0.33223992])]
    st_points = [None, None, None, array([0.347, 0.00733336]), None, None, None]

    s = []
    x = []
    o = []
    for i in range(N_cases):

        #eq = load_testing_equilibrium(i)

        print("Reading {}".format(gfiles[i]))

        eq = read_geqdsk(gfiles[i])
        # plt.figure()
        # eq._plot_overview()
        #plot_extremes(eq)

        assert np.allclose(eq._mg_axis, o_points[i])
        if eq._x_point is not None:
            assert np.allclose(eq._x_point, x_points[i])
        if eq._strike_point is not None:
            assert np.allclose(eq._strike_point, st_points[i])

        print('idx = {}'.format(i))
        print('mg axis = {}'.format(eq._mg_axis))
        print('x point = {}'.format(eq._x_point))
        print('strike point = {}'.format(eq._strike_point))


    #plt.show()
