# load OMAS package
import omas
from pleque.io import omas as plomas
import matplotlib.pyplot as plt


# load data from a pulse chosen from the ITER scenario database
ods = omas.load_omas_iter_scenario(pulse=131034, run=0)
eq = plomas.read(ods)

eq.plot_overview(x)
plt.show()