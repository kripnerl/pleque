import pytest
import pleque.tests.utils as test_util


@pytest.fixture(scope="module",
                params=[0, 1, 2, 3, 4, 5])
def equilibrium(request):
    yield test_util.load_testing_equilibrium(request.param)

@pytest.fixture(scope="module",
                params=[0, 1, 2, 3, 4, 5])
def geqdsk_file(request):
    yield test_util.get_test_equilibria_filenames()[request.param]