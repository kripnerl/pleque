import xarray as xr


@xr.register_dataset_accessor('pleque_coords')
class CoordsAccessor:
    def __init__(self, xarray_dataset):
        self._ds = xarray_dataset

    def foo(self):
        print('Help!!!')