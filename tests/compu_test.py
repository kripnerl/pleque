def test_compu():
    from pleque.io.compass import read_fiesta_equilibrium
    import pkg_resources

    resource_package = "pleque"
    gfile = 'resources/baseline_eqdsk'
    gfile = pkg_resources.resource_filename(resource_package, gfile)

    eq = read_fiesta_equilibrium(gfile)
