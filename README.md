# PLEQUE - **PL**asma **EQU**ilibrium **E**njoyment module \[pleɪɡ\]
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://img.shields.io/github/license/mashape/apistatus.svg)
[![py3comp](https://img.shields.io/badge/py3-compatible-brightgreen.svg)](https://img.shields.io/badge/py3-compatible-brightgreen.svg)

Python module for the simple manipulation with the tokamak plasma equilibrium.
For more information see the documentation at https://pleque.readthedocs.io.

**Note:** The work is still in the early development stage and `pleque` probably contains bugs. You are very welcome to
put a your wishes, found bugs or any other comment as an issue. There also may occur minor changes in code structure 
before `0.1.0` release. 

## Getting Started

### Prerequisites

The prerequisites should be maintained by `pip`.  

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

### Installing

From PyPI (https://pypi.org/project/pleque/):
```bash
pip install --user pleque
```

or use the unstable experimental release (probably with more fixed bugs):
```bash
 pip install --user -i https://test.pypi.org/simple/ pleque
```

or clone/copy the github repository and run

```bash
git clone https://github.com/kripnerl/pleque.git
cd pleque
pip install --user .
```


## Examples

```python
from pleque.io import readers
import matplotlib as plt

eqdsk_filename = "path_to_my_gfile.gfile"
# Create instance of `Equilibrium` class
eq = readers.read_geqdsk(eqdsk_filename)

# plot simple overview of the equilibrium:
eq.plot_overview()
```

Some other examples can be found as notebooks in the `notebooks` folder or in
the `examples` directory. 

## Version

0.0.3b5

## Authors

* **Lukas Kripner** - [kripnerl](https://github.com/kripnerl)
* **Matěj Tomeš** - [Mateesek](https://github.com/MatejTomes)

See also the list of [contributors](https://github.com/kripnerl/pleque/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Related projects

* [FreeGS](https://github.com/bendudson/freegs) - Free boundary Grad-Shafranov solver in Python
* [OMFIT](https://gafusion.github.io/OMFIT-source/) is an integrated modeling and experimental data analysis software for magnetically confined thermonuclear fusion experiments. The goal of OMFIT is to enhance existing scientific workflows and enable new integrated modeling capabilities. To achieve these goals OMFIT adopts a bottom-up collaborative development approach.
* [OMAS](https://gafusion.github.io/omas/) (Ordered Multidimensional Array Structure) is a Python library designed to simplify the interface of third-party codes with the ITER Integrated Modeling and Analysis Suite (IMAS) . ITER IMAS defines a data model, a data get/put API, and a data storage infrastructure used for manipulating ITER data.

## References
* [Sauter, O. & Medvedev, S. Y. "Tokamak coordinate conventions: COCOS." Comput. Phys. Commun. **184**, 293–302 (2013).](https://www.sciencedirect.com/science/article/pii/S0010465512002962)
* S. Jardin "Computational Methods in Plasma Physics" CRC Press.
