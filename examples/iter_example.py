# load OMAS package
import omas
from pleque.io import omas as plomas
import matplotlib.pyplot as plt

# load data from a pulse chosen from the ITER scenario database
ods = omas.load_omas_iter_scenario(pulse=130501, run=1)

# To save and load one can use pikle (or other OMAS io option):
# omas.save_omas_pkl(ods, "/home/kripner/Projects/pleque/notes/iter130501.pkl")
# ods = omas.load_omas_pkl("/home/kripner/Projects/pleque/notes/iter130501.pkl")

eq = plomas.read(ods, time=500)
eq.plot_overview()
plt.show()