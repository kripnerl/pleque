First steps
===========

.. highlight:: bash

Prerequisites
-------------

The following packages are required to install PLEQUE::

   python>=3.5
   numpy
   scipy
   shapely
   scikit-image
   xarray
   pandas
   h5py
   omas

They should be automatically handled by pip further in the installation process.

Download the source code
------------------------

First, pick where you wish to install the code::

  cd /desired/path/

There are two options how to get the code: from PyPI or by cloning the repository.

From PyPI (https://pypi.org/project/pleque/)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

   pip install --user pleque

Alternatively, you may use the unstable experimental release (probably with more fixed bugs)::

   pip install --user -i https://test.pypi.org/simple/ pleque

Clone the github repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

   git clone https://github.com/kripnerl/pleque.git
   cd pleque
   pip install --user .

Congratulations, you have just installed pleque!
