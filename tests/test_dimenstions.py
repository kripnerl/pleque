"""
Test module for checking the dimensions of arrays returned by functions in the Equilibrium class.

This module contains tests to verify that functions in the Equilibrium class that return
ndarrays have the correct shape according to the project requirements:

1. Vector functions should return arrays with shape:
   - (n_dim, n_elements) when grid=False
   - (n_dim, n_z, n_r) when grid=True

2. Scalar functions should return arrays with shape:
   - (n_elements) when grid=False
   - (n_z, n_r) when grid=True

The tests use a single equilibrium fixture for testing all functions.
"""

import numpy as np
import pytest

from pleque import Equilibrium


def get_scalar_function_names():
    """Helper function to get all scalar function names."""
    scalar_functions = []
    for attr_name in dir(Equilibrium):
        attr = getattr(Equilibrium, attr_name)
        if callable(attr) and hasattr(attr, "_scalar_function"):
            scalar_functions.append(attr_name)
    return scalar_functions


def get_vector_function_names():
    """Helper function to get all vector function names."""
    vector_functions = []
    for attr_name in dir(Equilibrium):
        attr = getattr(Equilibrium, attr_name)
        if callable(attr) and hasattr(attr, "_vector_function"):
            vector_functions.append(attr_name)
    return vector_functions


@pytest.fixture(params=get_scalar_function_names(),
                ids=lambda attr_name: attr_name)
def scalar_registered_functions(request, equilibrium):
    attr_name = request.param
    return getattr(equilibrium, attr_name)


@pytest.fixture(params=get_vector_function_names(),
                ids=lambda attr_name: attr_name)
def vector_registered_function(request, equilibrium):
    attr_name = request.param
    return getattr(equilibrium, attr_name)


@pytest.mark.parametrize("r, z, grid, exp_shape", (
        (np.linspace(1.0, 2.0, 5), np.linspace(-0.5, 0.5, 5), False, (5,)),
        (np.linspace(1.0, 2.0, 5), np.linspace(-0.5, 0.5, 7), True, (7, 5)),
        (*np.meshgrid(np.linspace(1.0, 2.0, 5), np.linspace(-0.5, 0.5, 7)), False, (7, 5))
))
def test_vector_function_dimensions(equilibrium, vector_registered_function, r, z, grid, exp_shape):
    """
    Test that vector functions return arrays with the correct shape:
    - (n_elements, n_dim) when grid=False
    - (n_dim, n_z, n_r) when grid=True
    
    According to the project requirements, functions returning vector quantities
    should have a shape of (n_elements, n_dim) when grid=False, where:
    - n_elements is the number of points
    - n_dim is the dimensionality of the vector (3 for 3D vectors)
    
    When grid=True, the shape should be (n_dim, n_z, n_r), where:
    - n_z is the number of Z coordinates
    - n_r is the number of R coordinates
    - n_dim is the dimensionality of the vector (3 for 3D vectors)
    
    """
    func = vector_registered_function
    result = func(R=r, Z=z, grid=grid)
    assert result.shape == (func._vector_ndim,
                            *exp_shape), f"Function {func.__name__} with grid=True returned shape {result.shape}, expected {(func._vector_ndim, *exp_shape)}"


@pytest.mark.parametrize("r, z, grid, exp_shape", (
        (np.linspace(1.0, 2.0, 5), np.linspace(-0.5, 0.5, 5), False, (5,)),
        (np.linspace(1.0, 2.0, 5), np.linspace(-0.5, 0.5, 7), True, (7, 5)),
        (*np.meshgrid(np.linspace(1.0, 2.0, 5), np.linspace(-0.5, 0.5, 7)), False, (7, 5))
))
def test_scalar_function_dimensions(equilibrium, scalar_registered_functions, r, z, grid, exp_shape):
    """
    Test that scalar functions return arrays with the correct shape:
    - (n_elements) when grid=False
    - (n_z, n_r) when grid=True
    
    According to the project requirements, functions returning scalar quantities
    should have a shape of (n_elements) when grid=False, where:
    - n_elements is the number of points
    
    When grid=True, the shape should be (n_z, n_r), where:
    - n_z is the number of Z coordinates
    - n_r is the number of R coordinates
    """
    func = scalar_registered_functions
    result = func(R=r, Z=z, grid=grid)
    assert result.shape == exp_shape, f"Function {func.__name__} with grid=True returned shape {result.shape}, expected {exp_shape}"
