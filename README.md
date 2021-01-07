# PLEQUE - **PL**asma **EQU**ilibrium **E**njoyment module \[pleɪɡ\]
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://img.shields.io/github/license/mashape/apistatus.svg)
[![py3comp](https://img.shields.io/badge/py3-compatible-brightgreen.svg)](https://img.shields.io/badge/py3-compatible-brightgreen.svg)

PLEQUE is a Python module allowing simple visualisation and manipulation of tokamak plasma equilibria.
For more information see the documentation at https://pleque.readthedocs.io.

**Note:** The work is still in the early development stage, so `pleque` probably contains bugs. You are very welcome to submit your wishes, encountered bugs or any other comments as an issue. Minor changes in the code structure may occur before the `0.1.0` release. 

## Getting Started

### Prerequisites

The following packages are required to install `pleque`:

```
python>=3.5
numpy
scipy
shapely
scikit-image
xarray
pandas
h5py
omas
```
They should be automatically handled by `pip` further in the installation process.  

### Download the source code

 First, pick where you wish to install the code:
```bash
 cd /desired/path/
```

There are two options how to get the code: from PyPI or by cloning the repository.


#### From PyPI (https://pypi.org/project/pleque/)
```bash
pip install --user pleque
```
Alternatively, you may use the unstable experimental release (probably with more fixed bugs):
```bash
 pip install --user -i https://test.pypi.org/simple/ pleque
```

#### Clone the github repository

```bash
git clone https://github.com/kripnerl/pleque.git
cd pleque
pip install --user .
```
 Congratulations, you have just installed `pleque`!

## Examples

The following example shows how to load an equilibrium saved in the `eqdsk` format. The equilibrium used here comes from a FIESTA simulation of the COMPASS-Upgrade tokamak.

```python
from pleque.io import readers
import pkg_resources
import matplotlib as plt

#Locate a test equilibrium
filepath = pkg_resources.resource_filename('pleque', 'resources/baseline_eqdsk')
```
The heart of `pleque` is its `Equilibrium` class, which contains all the equilibrium information (and much more). Typically its instances are called `eq`.

```python
# Create an instance of the `Equilibrium` class
eq = readers.read_geqdsk(filepath)
```
The `Equilibrium` class comes with tons of interesting functions and caveats.

```python
# Plot a simple overview of the equilibrium
eq.plot_overview()

# Calculate the separatrix area
sep_area = eq.lcfs.area

# Get absolute magnetic field magnitude at given point
R = 0.7 #m
Z = 0.1 #m
B = eq.B_abs(R, Z)
```

Equilibria may be visualised in many different ways; they may be used for mapping or field line tracing; the possibilities are virtually endless. If there's a caveat you find missing from `pleque`, write to us! Further examples can be found as notebooks in the `notebooks` folder or in the `examples` directory. 

## Version

0.0.5

## Authors

* **Lukáš Kripner** - [kripnerl](https://github.com/kripnerl)
* **Matěj Tomeš** - [Mateesek](https://github.com/MatejTomes)

See also the list of [contributors](https://github.com/kripnerl/pleque/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related projects

* [FreeGS](https://github.com/bendudson/freegs) - free boundary Grad-Shafranov solver in Python.
* [OMFIT](https://gafusion.github.io/OMFIT-source/) is an integrated modeling and experimental data analysis software for magnetically confined thermonuclear fusion experiments. The goal of OMFIT is to enhance existing scientific workflows and enable new integrated modeling capabilities. To achieve these goals OMFIT adopts a bottom-up collaborative development approach.
* [OMAS](https://gafusion.github.io/omas/) (Ordered Multidimensional Array Structure) is a Python library designed to simplify the interface of third-party codes with the ITER Integrated Modeling and Analysis Suite (IMAS) . ITER IMAS defines a data model, a data get/put API, and a data storage infrastructure used for manipulating ITER data.

## References
* [O. Sauter and S. Yu. Medvedev: *Tokamak coordinate conventions: COCOS*, Computer Physics Communications **184**, 293–302 (2013)](https://www.sciencedirect.com/science/article/pii/S0010465512002962)
* S. Jardin: *Computational Methods in Plasma Physics*, CRC Press
