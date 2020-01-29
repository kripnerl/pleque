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

Since `pleque` is all about tokamak equilibria, to try it out you first need to procure an equilibrium. Out of the several possible equilibrium formats, the following example uses the `eqdsk` format.

```python
from pleque.io import readers
import matplotlib as plt

# Create an instance of the `Equilibrium` class
eq = readers.read_geqdsk("path_to_my_gfile.gfile")
```
The heart of `pleque` is its `Equilibrium` class, whose instances are respective equilibria with tons of interesting functions and caveats on top.

```python
# Plot a simple overview of the equilibrium
eq.plot_overview()
```

Equilibria may be visualised in many different ways; they may be used for mapping or field line tracing; the possibilities are virtually endless. If there's a caveat you find missing from `pleque`, write to us! Further examples can be found as notebooks in the `notebooks` folder or in the `examples` directory. 

## Version

0.0.3

## Authors

* **Lukáš Kripner** - [kripnerl](https://github.com/kripnerl)
* **Matěj Tomeš** - [Mateesek](https://github.com/MatejTomes)

See also the list of [contributors](https://github.com/kripnerl/pleque/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related projects

* [FreeGS](https://github.com/bendudson/freegs) - Free boundary Grad-Shafranov solver in Python
* [OMFIT](https://gafusion.github.io/OMFIT-source/) is an integrated modeling and experimental data analysis software for magnetically confined thermonuclear fusion experiments. The goal of OMFIT is to enhance existing scientific workflows and enable new integrated modeling capabilities. To achieve these goals OMFIT adopts a bottom-up collaborative development approach.
* [OMAS](https://gafusion.github.io/omas/) (Ordered Multidimensional Array Structure) is a Python library designed to simplify the interface of third-party codes with the ITER Integrated Modeling and Analysis Suite (IMAS) . ITER IMAS defines a data model, a data get/put API, and a data storage infrastructure used for manipulating ITER data.

## References
* [Sauter, O. & Medvedev, S. Y. "Tokamak coordinate conventions: COCOS." Comput. Phys. Commun. **184**, 293–302 (2013)](https://www.sciencedirect.com/science/article/pii/S0010465512002962)
* S. Jardin "Computational Methods in Plasma Physics" CRC Press
