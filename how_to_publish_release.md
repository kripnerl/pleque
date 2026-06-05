# How to publish release:

1. Create new branch with name `release/[release_number]
    - `release_number` format: `major.minor.subminor` (e.g. 0.1.2) for main versions, or
    - `major.minor.subminor.patchnumber` if needed, or
    - `major.minor.subminorbbetaversion` (e.g. 0.1.1b3) for testing release 
2. Set release number 
   1. in `README.md`,
   2. in `pleque/__init__.py`,
   3. minor release number in `docs/source/conf.py` (e.g. 0.1)
3. Create push request to master/develop on GitLab.
4. Wait for the approval.
5. Tag the commit as `v[release_number]`
6. Upload the release to pip: 
    1. run `bash` in `pleque` home (checkout to master/develop)
    2. Prepare distribution in `dist` directory:
        ```bash
        poetry build
        ```  
    3. Upload the distribution to PyPI (in dist/ directory must be only one release):
        1. Standard distribution:
        ```bash
        twine upload dist/*
        ```
        1. Testing distribution:
        ```bash
        twine upload --repository-url https://test.pypi.org/legacy/ dist/*
        ```
       - Before the first release, auth poetry with `poetry config pypi-token.pypi 'pypi-YOUR_TOKEN_HERE'`
    4. Try to download it and install it via `pip`.
7. Check whether the documentation on `readthedocs` is build without error. 
8. Breath normally. 
