# How to publish release:

1. Create new branch with name `release_[release_number]`
1. Set release number in `README.md`.
1. Set release number in `setup.py`
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
    1. Upload the distribution to PyPI:
        ```bash
        twine upload dist/*
        ```
    1. Try to download it and install it via `pip`.
1. Check whether the documentation on `readthedocs` is build without error. 
1. Breath normally. 
