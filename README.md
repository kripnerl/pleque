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
python>=3.11
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

### Development with uv

For a development checkout, install the locked runtime and development
dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run tests through the managed environment:

```bash
uv run pytest
```

The optional COMPASS integration can be installed with:

```bash
uv sync --group compass
```

## Examples

The following example shows how to load an equilibrium saved in the `eqdsk` format. The equilibrium used here comes from a FIESTA simulation of the COMPASS-Upgrade tokamak.

```python
from importlib import resources

from pleque.io import readers
import matplotlib as plt

#Locate a test equilibrium
filepath = resources.files('pleque').joinpath('resources', 'baseline_eqdsk')
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

## Array convention

Public evaluation functions use a component-first convention for vector quantities and
the same spatial shape as the requested coordinates for scalar quantities.

* Scalar functions evaluated at paired points return `[n_elements]`.
* Scalar functions evaluated on a grid return `[n_z, n_r]`, matching `np.meshgrid(R, Z)`.
* Vector functions return `[n_dim, ...]`, for example `[n_dim, n_elements]` for paired points
  and `[n_dim, n_z, n_r]` for grids.
* Passing mesh-shaped `R` and `Z` arrays with `grid=False` is treated as elementwise evaluation
  and preserves the mesh shape.

## Configuration

All tunable algorithm parameters (grid resolutions, solver tolerances, search heuristics, ...)
have built-in defaults that can be overridden by a `pleque.toml` file — in the working
directory, in the user config directory (`~/.config/pleque/` or `%APPDATA%\pleque\`), or as a
`[tool.pleque]` table in `pyproject.toml` — or by `PLEQUE_`-prefixed environment variables:

```toml
[flux_surfaces]
n_psi = 300

[lcfs]
search_grid_nr = 1000
```

See the [configuration documentation](https://pleque.readthedocs.io/en/latest/configuration.html)
for the file lookup order and the full settings reference.

## Authors

* **Lukáš Kripner** - [kripnerl](https://github.com/kripnerl)
* **Matěj Tomeš** - [Mateesek](https://github.com/MatejTomes)

See also the list of [contributors](https://github.com/kripnerl/pleque/graphs/contributors) who participated in this project.

## Reference

If you deem it appropriate, please refer to the PLEQUE package using the following publication:

* [Kripner, L., Tomeš, M., Urban, J., Grover, O., Ficker, O., Macúšová, E., Peterka, M., Krbec, J., Jaulmes, F., Cerovský, J., Fridrich, D., 2019. Towards the integrated analysis of tokamak plasma equilibria: PLEQUE, in: 46th EPS Conference on Plasma Physics, EPS 2019. European Physical Society.](https://lac913.epfl.ch/epsppd3/2019/pdf/P4.1033.pdf)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Old versions note

Although the systematic development of the project was intended, it endup bit organic. From version > 0.1.0 the code will be updated with breaking changes. Here is the list of historical versions, which may be required in some codes.

* 0.0.8 - Tagged master branch with the last change from 19-08-2024.
* 0.0.9 - Tagged develop branch with maintained back compatibility with 0.0.8. 
* 0.0.10 All the phd-related work merged the master + update of array ordering. This version may thus introduce breaking changes! 


## Related projects

* [FreeGS](https://github.com/bendudson/freegs) - free boundary Grad-Shafranov solver in Python.
* [OMFIT](https://gafusion.github.io/OMFIT-source/) is an integrated modeling and experimental data analysis software for magnetically confined thermonuclear fusion experiments. The goal of OMFIT is to enhance existing scientific workflows and enable new integrated modeling capabilities. To achieve these goals OMFIT adopts a bottom-up collaborative development approach.
* [OMAS](https://gafusion.github.io/omas/) (Ordered Multidimensional Array Structure) is a Python library designed to simplify the interface of third-party codes with the ITER Integrated Modeling and Analysis Suite (IMAS) . ITER IMAS defines a data model, a data get/put API, and a data storage infrastructure used for manipulating ITER data.

## References
* [O. Sauter and S. Yu. Medvedev: *Tokamak coordinate conventions: COCOS*, Computer Physics Communications **184**, 293–302 (2013)](https://www.sciencedirect.com/science/article/pii/S0010465512002962)
* S. Jardin: *Computational Methods in Plasma Physics*, CRC Press
