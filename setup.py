from setuptools import setup
import os

def readme():
    return open(os.path.join(os.path.dirname(__file__), 'README.md')).read()

setup(
    name='pleque',
    version='0.0.1',
    packages=['pleque_test', 'pleque', 'pleque.io', 'pleque.utils'],
    url='https://pleque.readthedocs.io',
    license='MIT',
    author='Lukas Kripner',
    author_email='kripner@ipp.cas.cz',
    description='Python module for an easy work with a tokamak plasma equilibrium.',
    long_description=readme(),
    long_description_content_type="text/markdown",
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


