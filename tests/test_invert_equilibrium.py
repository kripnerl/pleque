import numpy as np

from pleque import Equilibrium
from pleque.tests.utils import load_testing_equilibrium, synthetic_dataset, synthetic_test_wall


def create_test_equilibrium():
    """Create a simple test equilibrium for testing inversion."""
    eq = Equilibrium(synthetic_dataset(), first_wall=synthetic_test_wall())

    return eq

def test_invert_vertically():
    """Test inverting the equilibrium vertically."""
    eq = create_test_equilibrium()

    # Print information about the equilibrium
    print(f"eq._basedata: {eq._basedata}")
    print(f"eq._first_wall: {eq._first_wall}")
    print(f"eq._mg_axis: {eq._mg_axis}")
    print(f"eq._psi_lcfs: {eq._psi_lcfs}")
    print(f"eq._x_points: {eq._x_points}")
    print(f"eq._strike_points: {eq._strike_points}")
    print(f"eq._init_method: {eq._init_method}")
    print(f"eq._spline_order: {eq._spline_order}")
    print(f"eq._cocos: {eq._cocos}")
    print(f"eq._verbose: {eq._verbose}")

    # Invert vertically
    eq_inv = eq.invert_equilibrium(vertically=True)

    # Check that the psi values are inverted
    # Original psi at (R, Z) should equal inverted psi at (R, -Z)
    R_test = 1.5
    Z_test = 0.5

    psi_orig = eq.psi(R=R_test, Z=Z_test)
    psi_inv = eq_inv.psi(R=R_test, Z=-Z_test)

    assert np.isclose(psi_orig, psi_inv)

def test_invert_first_wall():
    """Test inverting the first wall."""
    eq = create_test_equilibrium()

    # Invert first wall
    eq_inv = eq.invert_equilibrium(first_wall=True)

    # Check that the first wall is inverted
    # The Z coordinates should be inverted
    z_min = eq._basedata.Z.min().item()
    z_max = eq._basedata.Z.max().item()

    for i in range(len(eq._first_wall)):
        assert np.isclose(
            eq._first_wall[i, 1],
            z_max + z_min - eq_inv._first_wall[i, 1]
        )

def test_invert_toroidal_field():
    """Test inverting the toroidal field."""
    eq = create_test_equilibrium()

    # Invert toroidal field
    eq_inv = eq.invert_equilibrium(toroidal_field=True)

    # Check that the toroidal field is inverted
    # The F function should be inverted
    psi_n_test = 0.5

    F_orig = eq.F(psi_n=psi_n_test)
    F_inv = eq_inv.F(psi_n=psi_n_test)

    assert np.isclose(F_orig, -F_inv)

    B0_orig = eq.B_tor(eq.magnetic_axis)
    B0_inv = eq_inv.B_tor(eq_inv.magnetic_axis)

    assert np.isclose(B0_orig, -B0_inv)

def test_invert_current():
    """Test inverting the current."""
    eq = create_test_equilibrium()

    # Invert current
    eq_inv = eq.invert_equilibrium(current=True)

    # Check that the current is inverted
    # The pprime and FFprime functions should be inverted
    psi_n_test = 0.5
    R_test = 1.5
    Z_test = 0.5

    B_R_orig = eq.B_R(R=R_test, Z=Z_test)
    B_R_inv = eq_inv.B_R(R=R_test, Z=Z_test)

    assert np.isclose(B_R_orig, -B_R_inv)

    B_Z_orig = eq.B_Z(R=R_test, Z=Z_test)
    B_Z_inv = eq_inv.B_Z(R=R_test, Z=Z_test)

    assert np.isclose(B_Z_orig, -B_Z_inv)

    pprime_orig = eq.pprime(psi_n=psi_n_test)
    pprime_inv = eq_inv.pprime(psi_n=psi_n_test)

    FFprime_orig = eq.FFprime(psi_n=psi_n_test)
    FFprime_inv = eq_inv.FFprime(psi_n=psi_n_test)

    assert np.isclose(pprime_orig, -pprime_inv)
    assert np.isclose(FFprime_orig, -FFprime_inv)

def test_multiple_inversions():
    """Test inverting multiple properties at once."""
    eq = create_test_equilibrium()

    # Invert multiple properties
    eq_inv = eq.invert_equilibrium(
        vertically=True,
        first_wall=True,
        toroidal_field=True,
        current=True
    )

    # Check that all properties are inverted
    R_test = 1.5
    Z_test = 0.5
    psi_n_test = 0.5

    # Check psi
    psi_orig = eq.psi(R=R_test, Z=Z_test)
    psi_inv = eq_inv.psi(R=R_test, Z=-Z_test)
    assert np.isclose(psi_orig, -psi_inv)

    assert np.isclose(eq.magnetic_axis.Z, -eq_inv.magnetic_axis.Z)

    # Check poloidal field
    B_R_orig = eq.B_R(R=R_test, Z=Z_test)
    B_R_inv = eq_inv.B_R(R=R_test, Z=-Z_test)
    # Double inversion (vertically and poloidal_field) => no sign change
    assert np.isclose(B_R_orig, B_R_inv)

    psi_n_test = np.linspace(0.0, 1.0, 10)
    # import matplotlib.pyplot as plt
    # plt.plot(psi_n_test, eq.F(psi_n=psi_n_test), label="F")
    # plt.plot(psi_n_test, eq_inv.F(psi_n=psi_n_test), label="F_inv")
    # plt.legend()
    # plt.show()

    assert np.isclose(eq.F0, -eq_inv.F0)

    # Check toroidal field
    F_orig = eq.F(psi_n=psi_n_test)
    F_inv = -eq_inv.F(psi_n=psi_n_test)
    np.testing.assert_allclose(F_orig, F_inv, rtol=1e-2, atol=1e-2)

    # Check current
    pprime_orig = eq.pprime(psi_n=psi_n_test)
    pprime_inv = eq_inv.pprime(psi_n=psi_n_test)
    np.testing.assert_allclose(pprime_orig, -pprime_inv)

    # Check first wall and limiter point
    z_min = eq._basedata.Z.min().item()
    z_max = eq._basedata.Z.max().item()
    for i in range(len(eq._first_wall)):
        assert np.isclose(
            eq._first_wall[i, 1],
            z_max + z_min - eq_inv._first_wall[i, 1]
        )
    assert np.isclose(eq._strike_points[0, 1], z_max + z_min - eq_inv._strike_points[0, 1])

def test_invert_existing_equilibrium():
    """Test inverting an existing equilibrium."""
    # Load an existing equilibrium
    eq = load_testing_equilibrium(case=3)  # Using case 3 (g13127.1050)

    # Invert vertically
    eq_inv = eq.invert_equilibrium(vertically=True)

    # Check that the psi values are inverted
    # Original psi at (R, Z) should equal inverted psi at (R, -Z)
    R_test = eq.magnetic_axis.R
    Z_test = 0.1  # Some non-zero Z value

    psi_orig = eq.psi(R=R_test, Z=Z_test)
    psi_inv = eq_inv.psi(R=R_test, Z=-Z_test)

    # Print the values for debugging
    print(f"psi_orig = {psi_orig}, psi_inv = {psi_inv}")

    # Check that the values have the same sign and are close in magnitude
    assert np.sign(psi_orig) == np.sign(psi_inv)
    assert np.abs((psi_orig - psi_inv) / psi_orig) < 0.1  # Allow up to 10% difference

    # Invert first wall
    eq_inv = eq.invert_equilibrium(first_wall=True)

    # Check that the first wall is inverted
    # The Z coordinates should be inverted
    z_min = eq._basedata.Z.min().item()
    z_max = eq._basedata.Z.max().item()

    for i in range(len(eq._first_wall)):
        assert np.isclose(
            eq._first_wall[i, 1],
            z_max + z_min - eq_inv._first_wall[i, 1]
        )

    # Invert toroidal field
    eq_inv = eq.invert_equilibrium(toroidal_field=True)

    # Check that the toroidal field is inverted
    # The F function should be inverted
    psi_n_test = 0.5

    F_orig = eq.F(psi_n=psi_n_test)
    F_inv = eq_inv.F(psi_n=psi_n_test)

    # Print the values for debugging
    print(f"F_orig = {F_orig}, F_inv = {F_inv}")

    # Check that the values have opposite signs
    assert np.sign(F_orig) == -np.sign(F_inv)
    # Check that the magnitudes are close
    assert np.isclose(np.abs(F_orig), np.abs(F_inv))

    # Invert current
    eq_inv = eq.invert_equilibrium(current=True)

    # Check that the current is inverted
    # The pprime and FFprime functions should be inverted
    pprime_orig = eq.pprime(psi_n=psi_n_test)
    pprime_inv = eq_inv.pprime(psi_n=psi_n_test)

    FFprime_orig = eq.FFprime(psi_n=psi_n_test)
    FFprime_inv = eq_inv.FFprime(psi_n=psi_n_test)

    assert np.isclose(pprime_orig, -pprime_inv)
    assert np.isclose(FFprime_orig, -FFprime_inv)
