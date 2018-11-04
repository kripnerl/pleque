from setuptools import setup

setup(
    name='pleque',
    version='0.0.1',
    packages=['pleque_test', 'pleque', 'pleque.io', 'pleque.utils'],
    url='https://pleque.readthedocs.io',
    license='MIT',
    author='Lukas Kripner',
    author_email='kripner@ipp.cas.cz',
    description='Python module for easy work with a tokamak plasma equilibrium.',
    install_requires=[
        'numpy',
        'scipy',
        'shapely',
        'scikit-image',
        'xarray',
        'pandas',
    ],
    zip_safe=False
)


