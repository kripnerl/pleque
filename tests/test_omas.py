import numpy as np
import os

user = os.environ.get("USER", "dummy_user")
os.environ["USER"] = user

# omas = pytest.importorskip("omas")

from pleque.io import omas as plomas
from pleque.tests.utils import load_testing_equilibrium

# load the equilibrium data

eq = load_testing_equilibrium()
grid_2d = eq.grid(resolution=[1e-3, 1e-3], dim="step")
grid_1d = eq.coordinates(psi_n=np.linspace(0, 1, 100))
gridtype = 1

# ods = plomas.write(eq)
ods = plomas.write(eq, grid_1d, grid_2d)
ods = plomas.write(eq, grid_1d, grid_2d, gridtype=1, ods=ods)

eq2 = plomas.read(ods)

import matplotlib.pyplot as plt
eq2.plot_overview()

plt.show()