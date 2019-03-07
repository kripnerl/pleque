# How to publish release:

1. Create new branch with name `release/[release_number]
    - `release_number` format: `major.minor.subminor` (e.g. 0.1.2) for main versions, or
    - `major.minor.subminor.patchnumber` if needed, or
    - `major.minor.subminorbbetaversion` (e.g. 0.1.1b3) for testing release 
1. Set release number in `README.md`.
1. Set release number in `pleque/__init__.py`
1. Set release number in `docs/source/conf.py`
1. Create push request to master on GitLab.
1. Wait for the approval.
1. Tag the commit as `v[release_number]`
1. Upload the release to pip: 
    1. run `bash` in `pleque` home (checkout to master)
    1. Prepare distribution in `dist` directory:
        ```bash
        python setup.py sdist bdist_wheel 
        ```  
    1. Upload the distribution to PyPI (in dist/ directory must be only one release):
        1. Standard distribution:
        ```bash
        twine upload dist/*
        ```
        1. Testing distribution:
        ```bash
        twine upload --repository-url https://test.pypi.org/legacy/ dist/*
        ```
    1. Try to download it and install it via `pip`.
1. Check whether the documentation on `readthedocs` is build without error. 
1. Breath normally. 
