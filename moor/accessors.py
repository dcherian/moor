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

        self._splitseasons = {'NE': [12, 1, 2],
                              'Mar': [3],
                              'Apr': [4],
                              'SW': [5, 6, 7, 8, 9],
                              'SWNE': [10, 11]}

        self._labels = np.asarray(['    '] * len(xarray_obj))
        self._splitlabels = np.asarray(['    '] * len(xarray_obj))

        for ss in self._seasons:
            self._labels[self._obj.time.dt.month.isin(
                self._seasons[ss]).values] = ss

        for ss in self._splitseasons:
            self._splitlabels[self._obj.time.dt.month.isin(
                self._splitseasons[ss]).values] = ss

    @property
    def labels(self):
        obj_type = type(self._obj)
        return obj_type(self._labels, name='monsoon', coords=self._obj.coords,
                        dims=self._obj.dims)

    @property
    def splitlabels(self):
        obj_type = type(self._obj)
        return obj_type(self._splitlabels,
                        name='monsoon',
                        coords=self._obj.coords,
                        dims=self._obj.dims)
