from setuptools import setup, find_packages
import os

def readme():
    return open(os.path.join(os.path.dirname(__file__), 'README.md')).read()

setup(
    name='pleque',
    version='0.0.3b5',
    packages=find_packages(),  # ['pleque', 'pleque.test', 'pleque.io', 'pleque.utils'],
    package_data={'pleque': ['resources/*']},
    url='https://pleque.readthedocs.io',
    download_url='https://github.com/kripnerl/pleque',
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
        'scikit-image>=0.14.2',
        'xarray',
        'pandas',
        'h5py',
        'omas',
    ],
    zip_safe=False
)


