import xarray as xr
import numpy as np


@xr.register_dataarray_accessor('monsoon')
class MonsoonAccessor(object):

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._seasons = {'NE': [1, 2, 3],
                         'NESW': [4, 5],
                         'SW': [6, 7, 8, 9],
                         'SWNE': [10, 11, 12]}

        names = np.asarray(list(self._seasons.keys()))

        inds = np.zeros(self._obj.time.shape)
        for index, ss in enumerate(self._seasons):
            inds[np.in1d(self._obj.time.dt.month,
                         np.asarray(self._seasons[ss]))] = index

        self._labels = names[inds.astype(np.int32)]

    def _monsoon_func(self, label):
        obj_type = type(self._obj)

        return obj_type(self._labels[np.where(self._labels == label.upper())],
                        name='monsoon', coords=self._coords,
                        dims=self._obj.dims)

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
