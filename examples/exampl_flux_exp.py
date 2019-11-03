# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 23:34:45 2019

@author: Ondrej
"""
import sys
sys.path.append('C:\\Users\\Ondrej\\Projects\\pleque\\')

import matplotlib.pyplot as plt

from pleque.tests.utils import load_testing_equilibrium
from pleque.core.coordinates import Coordinates
import numpy as np
from scipy.interpolate import splprep, splev
eq = load_testing_equilibrium(4)

surf_lcfs = eq._flux_surface(psi_n=1 - 1e-6)[0]
first_wall=eq.first_wall


fign,axn=plt.subplots()
axn.set_aspect('equal')
cmap=plt.get_cmap('jet')


fw = eq.first_wall


ratio=eq.outter_parallel_fl_expansion_coef(fw)


for i in range(0,npoints-1):
    s=ax.scatter(fw.R, newpoints[1][i], c = cmap((impcos[i]-np.amin(impcos))/(np.amax(impcos)-np.amin(impcos))))


plt.show()