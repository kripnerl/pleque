def test_compu():
    from pleque.io.compass import read_fiesta_equilibrium
    from pleque.tests.utils import resource_path

    resource_package = "pleque"
    gfile = 'resources/baseline_eqdsk'
    gfile = resource_path(resource_package, gfile)

    eq = read_fiesta_equilibrium(gfile)
