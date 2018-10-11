import numpy as np

import xarray as xr


@xr.register_dataarray_accessor('monsoon')
class MonsoonAccessor(object):

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._seasons = {'NE': [12, 1, 2],
                         'NESW': [3, 4],
                         'SW': [5, 6, 7, 8, 9],
                         'SWNE': [10, 11]}

        self._labels = np.asarray(['    '] * len(xarray_obj))

        for ss in self._seasons:
            self._labels[self._obj.time.dt.month.isin(
                self._seasons[ss]).values] = ss

    def _monsoon_func(self, label):
        obj_type = type(self._obj)

        return obj_type(self._obj.where(self._labels == label.upper()),
                        name=self._obj.name + ' | ' + label.upper(),
                        coords=self._obj.coords, dims=self._obj.dims)

    @property
    def ne(self):
        return self._monsoon_func('ne')

    @property
    def nesw(self):
        return self._monsoon_func('nesw')

    @property
    def sw(self):
        return self._monsoon_func('sw')

    @property
    def swne(self):
        return self._monsoon_func('swne')

    @property
    def labels(self):
        obj_type = type(self._obj)
        return obj_type(self._labels, name='monsoon', coords=self._obj.coords,
                        dims=self._obj.dims)
