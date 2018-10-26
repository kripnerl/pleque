from .testing_utils import get_test_equilibria

def test_q_profiles():
    from pleque.io.compass import read_fiesta_equilibrium
    from pleque.io._geqdsk import read as read_gfile

    import matplotlib.pyplot as plt
    import numpy as np

    g_file = get_test_equilibria()[0]

    eq = read_fiesta_equilibrium(g_file)
    with open(g_file, 'r') as f:
        eq_dict = read_gfile(f)

    print(eq_dict.keys())

    q_1 = []
    q_2 = []
    psi_ns = []
    NN = eq_dict['nx']*4
    for i in range(1, NN):
        psi_n = i/NN
        psi_ns.append(psi_n)
        surf = eq.flux_surface(psi_n=psi_n, resolution=[1e-3, 1e-3])
        q_1.append(surf[0].eval_q)
        q_2.append(surf[0].eval_q_2)

    q_1 = np.array(q_1)
    q_2 = np.array(q_2)

    plt.figure()
    plt.plot(np.linspace(0, 1, eq_dict['nx']), eq_dict['qpsi'], 'o--', label='gfile')
    plt.plot(psi_ns, q_1, 'x-', label='calculated')
    plt.plot(psi_ns, q_2, '+--', label='calculated')
    plt.legend()

    plt.show()