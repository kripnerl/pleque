from pleque.io.compass import read_efithdf5
from os.path import  expanduser

eq = read_efithdf5(expanduser("~/EFIT/17636.1.h5"), time=1125)
print(eq)
