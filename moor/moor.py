import warnings

import airsea
import dcpy.oceans
import dcpy.plots
import dcpy.ts
import dcpy.util
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seawater as sw
import xfilter
import xrft
from dcpy.util import mdatenum2dt64

import xarray as xr


def _get_dt_in_days(time):
    return (time.diff(dim='time').median().values
            / np.timedelta64(1, 'D'))


def _decode_time(t0, t1):
    '''
    Utility function to decode time ranges.
    '''

    return (pd.to_datetime(t0), pd.to_datetime(t1))


def _corner_label(label: str, x=0.95, y=0.9, ax=None, alpha=0.05):
    '''
    Adds label to a location specified by (x,y).

    Input:
        (x,y) : location in normalized axis co-ordinates
                default: (0.95, 0.9)
        label : string
    '''

    if ax is None:
        ax = plt.gca()

    ax.text(x, y, label,
            color='k',
            horizontalalignment='right',
            verticalalignment='center',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor=None))


def _colorbar2(mappable):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    try:
        ax = mappable.axes
    except AttributeError:
        ax = mappable.ax

    fig = ax.figure

    # http://joseph-long.com/writing/colorbars/ and
    # https://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3%", pad=0.05)

    return fig.colorbar(mappable, cax=cax)


def _colorbar(hdl, ax=None, format='%.2f', **kwargs):

    if ax is None:
        try:
            ax = hdl.axes
        except AttributeError:
            ax = hdl.ax

    # if not isinstance(ax, list):
    #     box = ax.get_position()

    # if isinstance(ax, list) and len(ax) > 2:
    #     raise ValueError('_colorbar only supports passing 2 axes')

    # if isinstance(ax, list) and len(ax) == 2:
    #     box = ax[0].get_position()
    #     box2 = ax[1].get_position()

    #     box.y0 = np.min([box.y0, box2.y0])
    #     box.y1 = np.max([box.y1, box2.y1])

    #     box.y0 += box.height * 0.05
    #     box.y1 -= box.height * 0.05

    # axcbar = plt.axes([(box.x0 + box.width) * 1.02,
    #                    box.y0, 0.01, box.height])
    # hcbar = plt.gcf().colorbar(hdl, cax=axcbar,
    #                            format=mpl.ticker.FormatStrFormatter(format),
    #                            ticks=mpl.ticker.MaxNLocator('auto'),
    #                            pad=0)

    # supports constrained_layout
    hcbar = plt.gcf().colorbar(hdl, ax=ax, pad=0,
                               format=mpl.ticker.FormatStrFormatter(format),
                               ticks=mpl.ticker.MaxNLocator('auto'),
                               **kwargs)

    return hcbar


class moor:
    ''' Class for a *single* mooring that has χpods '''

    def __init__(self, lon, lat, name, short_name, kind, datadir):

        import collections

        self.name = name
        self.short_name = short_name
        self.kind = kind
        self.datadir = datadir

        # location
        self.lon = lon
        self.lat = lat
        self.inertial = xr.DataArray(
            1 / (2 * np.pi / (dcpy.oceans.coriolis(self.lat) * 86400)))
        self.inertial.attrs['units'] = 'cpd'
        self.inertial.attrs['long_name'] = 'Inertial frequency'

        self.AddSeason()
        self.events = dict()
        self.deployments = []
        self.deploy = dict()

        self.ctd = xr.Dataset()  # TAO CTD
        self.met = xr.Dataset()  # TAO met
        self.flux = xr.Dataset()
        self.tropflux = xr.Dataset()  # tropflux
        self.vel = xr.Dataset()
        self.sst = xr.Dataset()
        self.niw = xr.Dataset()
        self.ssh = []

        # chipods
        self.χpod = collections.OrderedDict()
        self.zχpod = dict([])

        self.turb = xr.Dataset()
        # combined turb data
        self.pitot = xr.Dataset()

    @property
    def χ(self):
        return self.turb.χ

    @property
    def ε(self):
        return self.turb.ε

    @property
    def KT(self):
        return self.turb.KT

    @property
    def Jq(self):
        return self.turb.Jq

    @property
    def KS(self):
        return self.turb.KS

    @property
    def Js(self):
        return self.turb.Js

    @property
    def N2(self):
        return self.turb.N2

    @property
    def Tz(self):
        return self.turb.Tz

    @property
    def Sz(self):
        return self.turb.Sz

    def __repr__(self):

        podstr = ''
        for unit in self.χpod:
            pod = self.χpod[unit]
            podstr += '\t' + pod.name[2:]
            times = (pd.to_datetime(pod.time[0].values).strftime('%Y-%b-%d')
                     + ' → '
                     + pd.to_datetime(pod.time[-2].values).strftime('%Y-%b-%d'))
            podstr += ' | ' + times + '\n'

        if self.kind == 'ebob':
            zz = np.percentile(self.zχpod, [10, 50, 90], axis=1).T
            podstr += '\nzχpod: \t' + np.array2string(zz, precision=1,
                                                      prefix='        ') + '\n'

        specstr = ''
        for ss in self.events:
            specstr += ('\t' + ss + ' | '
                        + self.events[ss][0].strftime('%Y-%b-%d')
                        + ' → '
                        + self.events[ss][1].strftime('%Y-%b-%d')
                        + '\n')

        specstr = specstr[1:]  # remove initial tab

        return ('mooring ' + self.name
                + '\nχpods: ' + podstr
                + '\nEvents: ' + specstr)

    def __str__(self):
        return self.name + ' mooring'

    def monin_obukhov(self):

        g = 9.81
        ρ0 = 1025
        cp = 4200
        α = 1.7e-4
        k = 0.41

        # Thorpe (4.1)
        B = g * α * self.flux.Jq0.where(self.flux.Jq0 < 0) / ρ0 / cp

        ustar = np.sqrt(self.met.τ / ρ0)

        Bi = B.interp(time=ustar.time)

        Lmo = -ustar**3 / k / Bi
        Lmo.name = 'Lmo'

        Lmo[abs(Lmo) > 200] = np.nan

        return Lmo

    def CombineTurb(self):
        '''
        Combines all χpod χ, ε, KT, Jq etc. into a single DataArray each
        '''

        z = []
        pitot_shear = []
        pitot_spd = []

        t = []
        for idx, unit in enumerate(self.χpod):
            pod = self.χpod[unit]
            t.append(pod.turb[pod.best]['time'])

        tall = pd.Series(np.array([[np.nanmin(tt), np.nanmax(tt)] for tt in t])
                         .ravel()).dt.round('D')
        tcommon = pd.date_range(tall.min(), tall.max(), freq='10min').values

        interpargs = {'time': tcommon}
        estimates = []
        for idx, unit in enumerate(self.χpod):
            pod = self.χpod[unit]

            timevec = pod.turb[pod.best]['time']

            z.append(xr.DataArray(pod.depth * np.ones((len(timevec), 1)),
                                  dims=['time', 'depth'],
                                  coords={'time': timevec,
                                          'depth': [np.nanmedian(pod.depth)]})
                     .interp(**interpargs))

            ρ1 = sw.pden(pod.ctd1.S, pod.ctd1.T, pod.ctd1.z)
            ρ2 = sw.pden(pod.ctd2.S, pod.ctd2.T, pod.ctd2.z)
            ρ = (xr.DataArray(
                np.nanmean(np.array([ρ1, ρ2]), axis=0),
                dims=['time'],
                coords={'time': dcpy.util.mdatenum2dt64(pod.ctd1.time)})
                .interp(**interpargs).expand_dims('depth'))

            T = (xr.DataArray(
                np.nanmean(np.array([pod.ctd1.T, pod.ctd2.T]), axis=0),
                dims=['time'],
                coords={'time': dcpy.util.mdatenum2dt64(pod.ctd1.time)})
                .interp(**interpargs).expand_dims('depth'))

            S = (xr.DataArray(
                np.nanmean(np.array([pod.ctd1.S, pod.ctd2.S]), axis=0),
                dims=['time'],
                coords={'time': dcpy.util.mdatenum2dt64(pod.ctd1.time)})
                .interp(**interpargs).expand_dims('depth'))

            mld = self.mld.interp(**interpargs)
            ild = self.ild.interp(**interpargs)

            coords = {'z': (['time', 'depth'], z[-1].values),
                      'time': tcommon,
                      'depth': [pod.depth],
                      'lat': self.lat,
                      'lon': self.lon,
                      'ρ': ρ,
                      'S': S,
                      'T': T,
                      'mld': mld,
                      'ild': ild}

            debug_combine = False

            est = pod.chi[pod.best].copy()
            dt = (est.time.dt.floor('min').diff('time').values
                  .astype('timedelta64[s]').astype('float32'))
            median_dt = np.median(dt)
            inds = np.where((dt - median_dt) > 3600)[0]
            time_old = pod.chi[pod.best].time.reset_coords(drop=True)

            if 'dzdT' in est:
                est = est.drop(['dzdT', 'no_min_dz', 'sgn', 'eps'])

            time_new = time_old
            # NaN out large gaps
            if inds.size > 0:
                for ii in inds:
                    t0 = time_old[ii].values
                    t1 = time_old[ii + 1].values

                    warnings.warn('Found large gap. NaNing out...')
                    gap = pd.date_range(t0, t1, freq=str(int(median_dt)) + 's')
                    gap = xr.DataArray(gap.values[1:], dims=['time'],
                                       coords={'time': gap.values[1:]},
                                       name='time')
                    if (((t1 - gap[-1].values)
                         .astype('timedelta64[s]').astype('float32'))
                        < median_dt):
                        gap = gap[:-1]

                    time_new = xr.concat([time_new.sel(time=slice(None, t0)),
                                          gap,
                                          time_new.sel(time=slice(t1, None))],
                                         dim='time')

            est = est.reindex({'time': time_new})

            est = est.rename({'Kt': 'KT',
                              'Ks': 'KS',
                              'dTdz': 'Tz',
                              'chi': 'χ'})

            if debug_combine:
                plt.figure()
                pod.chi[pod.best].Kt.plot.line(yscale='log', label='original')
                est.KT.plot.line(yscale='log', label='reindexed')
                plt.legend()

            if 'depth' in est:
                # ebob
                est = est.rename({'depth': 'z'})

            if 'w' in pod.best:
                # strided rolling mean is faster than resample
                nt = int(10 * 60 / np.median(dt))
                est = est.rename({'eps_Kt': 'ε'})
                if nt > 1:  # if 1 min wda estimate, average to 10 minute
                    est = (est
                           .rolling(time=nt, center=True)
                           .construct('chunk', stride=nt)
                           .mean('chunk'))
            else:
                est = (est.rename({'eps': 'ε'}))

            # interpolate to common time vector
            est = (est.interp(time=tcommon, method='nearest')
                   .expand_dims('depth'))

            # add in extra info
            for cc in coords:
                est[cc] = coords[cc]
                est = est.set_coords(cc)

            estimates.append(est)

            if debug_combine:
                est.KT.plot(label='interpolated', _labels=False)
                plt.title(pod.name + '|' + pod.best)

            # if pod.pitot is not None:
            #     pitot_spd.append(pod.pitot.spd.interp(time=tcommon)
            #                      .expand_dims('depth'))

            #     if 'shear' in pod.pitot:
            #         pitot_shear.append(pod.pitot.shear.interp(time=tcommon)
            #                            .expand_dims('depth'))

        def merge(x0):
            x = xr.merge(map(lambda xx: xx.to_dataset(), x0))[x0[0].name]
            x['depth'] = x.z.mean(dim='time')
            # grouped by depth
            a = [x.sel(depth=zz, drop=False) for zz in np.unique(x['depth'])]
            # concat depth groups in time
            # b = [ for xx in a]

            def merge2(aa):
                if aa.ndim > 3:
                    return xr.merge(
                        [aa.isel(depth=zz)
                         for zz in np.arange(len(np.atleast_1d(aa.depth)))])
                else:
                    return aa.to_dataset()

            b = [merge2(aa) for aa in a]
            return xr.concat(b, dim='depth')

        self.turb = xr.Dataset()
        self.turb = xr.merge([self.turb] + estimates)

        if pitot_shear != []:
            self.pitot['shear'] = merge(pitot_shear).shear
        if pitot_spd != []:
            self.pitot['spd'] = merge(pitot_spd).spd

        if self.kind == 'ebob':
            self.zχpod = (self.ctd.depth.isel(z=[0, 1]).interp(time=tcommon)
                          + xr.DataArray([5, 10], dims=['z']))

            if self.name == 'NRL2':
                self.zχpod = self.zχpod.isel(z=1).expand_dims('z')
                self.turb.z.values = self.zχpod.values.T
            else:
                self.turb.z.values = self.zχpod.values.T

            self.zχpod = self.zχpod.rename({'z': 'num'})
        else:
            self.zχpod = (self.turb.z.reset_coords().z
                          .rename({'depth': 'num'})
                          .transpose())

        if 'RAMA' in self.name:
            self.turb['tau'] = self.met.τ.interp(time=self.turb.time.values)
            for vv in ['uz', 'vz']:
                self.turb[vv] = xr.zeros_like(self.turb.tau)
                self.turb['wkb_' + vv] = xr.zeros_like(self.turb.tau)

        elif 'NRL' in self.name:
            self.turb['tau'] = (self.tropflux.tau
                                .drop(['latitude', 'longitude'])
                                .interp(time=self.turb.time.values))

            shear = self.interp_shear('bins')
            wkb_shear = self.interp_shear('bins', wkb_scale=True)

            for vv in ['uz', 'vz']:
                self.turb[vv] = (shear[vv]
                                 .interp(time=self.turb.time.values))
                self.turb['wkb_' + vv] = (wkb_shear[vv]
                                          .interp(time=self.turb.time.values))

        self.turb.uz.attrs['long_name'] = '$u_z$'
        self.turb.vz.attrs['long_name'] = '$v_z$'
        self.turb.uz.attrs['units'] = '1/s'
        self.turb.vz.attrs['units'] = '1/s'
        self.turb.wkb_uz.attrs['long_name'] = 'WKB $u_z$'
        self.turb.wkb_vz.attrs['long_name'] = 'WKB $v_z$'
        self.turb.wkb_uz.attrs['units'] = '1/s'
        self.turb.wkb_vz.attrs['units'] = '1/s'

        self.turb['wind_input'] = (self.niw.reset_coords().true_flux)
        self.turb['wind_input'].attrs['long_name'] = 'Local wind input $Π$'
        self.turb['wind_input'].attrs['units'] = 'W/m²'

        self.zχpod.num.values = (np.arange(self.zχpod.shape[0]) + 1)
        self.zχpod.attrs['long_name'] = 'χpod depth'
        self.zχpod.attrs['units'] = 'm'

        self.turb.KT.attrs['long_name'] = '$K_T$'
        self.turb.KT.attrs['units'] = 'm²/s'
        self.turb.KS.attrs['long_name'] = '$K_S$'
        self.turb.KS.attrs['units'] = 'm²/s'
        self.turb.ε.attrs['long_name'] = '$ε$'
        self.turb.ε.attrs['units'] = 'W/kg'
        self.turb.χ.attrs['long_name'] = '$χ$'
        self.turb.χ.attrs['units'] = 'm²/s'
        self.turb.Jq.attrs['long_name'] = '$J_q^t$'
        self.turb.Jq.attrs['units'] = 'W/m²'

        self.turb.Tz.attrs['units'] = 'C/m'

        # Estimating Sz as a difference between N2 and Tz is not a good idea
        # with WDA Tz estimate! Also N2 is estimated using pot density so
        # there's some difference actually
        # self.turb['Sz'] = (-(self.turb.N2 / 9.81 - 1.7e-4 * self.turb.Tz)
        #                   / 7.6e-4)
        self.turb.Sz.attrs['units'] = '1/m'

        self.turb['z'] = self.turb.z.transpose(*(self.turb.S.dims))
        # make sure max_KT filter is applied on KS too
        self.turb['KS'] = self.turb.KS.where(~np.isnan(self.turb.KT.values))
        self.turb['Js'] = - self.turb.ρ * self.turb.KS * self.turb.Sz
        self.turb.Js.attrs['long_name'] = '$J_s^t$'
        self.turb.Js.attrs['units'] = 'g/m²/s'

    def ReadSSH(self):
        ssha = xr.open_dataset('../datasets/ssh/' +
                               'dataset-duacs-rep-global-merged' +
                               '-allsat-phy-l4-v3_1522711420825.nc')

        ssha['EKE'] = np.hypot(ssha.ugosa, ssha.vgosa)

        self.ssh = ssha.sel(latitude=self.lat,
                            longitude=self.lon,
                            method='nearest').load()

    def ReadNIW(self):
        dirname = '../datasets/ewa/'

        self.niw = xr.open_dataset(dirname + self.name + '.nc').load()
        self.niw['latitude'] = self.lat
        self.niw['longitude'] = self.lon
        self.niw = self.niw.set_coords(['latitude', 'longitude'])
        self.niw.time.values = self.niw.time.dt.round('H')

    def ReadCTD(self, fname: str, FileType: str='ramaprelim'):

        from scipy.io import loadmat

        if FileType == 'ramaprelim':
            mat = loadmat(fname, squeeze_me=True, struct_as_record=False)
            try:
                mat = mat['rama']
            except KeyError:
                pass

            time = dcpy.util.mdatenum2dt64(mat.time - 366)
            z = np.floor(mat.depth)
            zmat = np.tile(z, (len(mat.time), 1)).T

            temp = xr.DataArray(mat.temp, dims=['depth', 'time'],
                                coords=[z, time], name='T')
            sal = xr.DataArray(mat.sal, dims=['depth', 'time'],
                               coords=[z, time], name='S')

            ρ = xr.DataArray(sw.pden(sal, temp, zmat),
                             dims=['depth', 'time'], coords=[z, time],
                             name='ρ')

            self.ctd = xr.merge([self.ctd, temp, sal, ρ])

        if FileType == 'rama':
            fname_ = fname + 't' + str(self.lat) + 'n' \
                + str(self.lon) + 'e' + '_10m.cdf'
            f = xr.open_dataset(fname_)
            self.ctd["T"] = f['T_20'].squeeze()

            fname_ = fname + 's' + str(self.lat) + 'n' \
                    + str(self.lon) + 'e' + '_hr.cdf'
            f = xr.open_dataset(fname_)
            self.ctd["S"] = f["S_41"].squeeze().interp(time=f.time.data)
            self.ctd["ρ"] = (self.ctd.S.dims, sw.pden(self.ctd.S, self.ctd.T, self.ctd.depth))

        if FileType == 'ebob':
            mat = loadmat(self.datadir + '/ancillary/ctd/' +
                          fname + 'SP-deglitched.mat', squeeze_me=True)
            temp = mat['temp']
            salt = mat['salt']
            pres = mat['pres']
            ρ = sw.pden(salt, temp, pres)

            mat2 = loadmat(self.datadir + '/ancillary/ctd/' +
                           'only_temp/EBOB_' + fname + '_WTMP.mat',
                           squeeze_me=True)
            temp2 = mat2['Wtmp' + fname[-1]].T

            time = dcpy.util.mdatenum2dt64(mat['time'] - 366)
            pres = np.float16(pres)
            z2 = mat2['dbar_dpth']

            self.ctd = xr.Dataset({'S': (['z', 'time'], salt),
                                   'T_S': (['z', 'time'], temp),
                                   'T': (['depth2', 'time'], temp2),
                                   'ρ': (['z', 'time'], ρ)},
                                  coords={'depth': (['z', 'time'], pres),
                                          'depth2': ('depth2', z2),
                                          'time': ('time', time[0, :])})
            self.ctd['depth'] = self.ctd.depth.fillna(0)

            try:
                Sgrid = xr.open_dataset(
                    '../intermediate-files/' + self.name + '-sgrid.nc')
                self.ctd['S_T'] = Sgrid.S.load()
                self.ctd['ρ_T'] = xr.apply_ufunc(
                    sw.pden, self.ctd.S_T, self.ctd.T, self.ctd.depth2)
            except FileNotFoundError:
                pass

            if fname == 'NRL3':
                # instrument is bad. All salinities are in 20s.
                # Simple offset correction doesn't help
                self.ctd.S.isel(z=3).values.fill(np.nan)

#    def calc_wind_flux(self):

    def calc_mld_ild_bld(self):
        def interp_to_1m(data):
            if np.any(np.isnan(data.values.T[:, 0])):
                data = data.bfill(dim='depth')

            f = sp.interpolate.RectBivariateSpline(data.time.astype('float32'),
                                                   data.depth,
                                                   data.values.T,
                                                   kx=1, ky=1, s=None)

            idepths = np.arange(data.depth.min(), data.depth.max() + 1, 1)
            datai = xr.DataArray(f(data.time.astype('float32'), idepths),
                                 dims=['time', 'depth'],
                                 coords=[data.time, idepths])
            return datai

        def find_mld(data, criterion):
            try:
                return data.depth[(np.abs(data - data.isel(depth=1)) >
                                   criterion).argmax(axis=1)].drop('depth')
            except AttributeError:
                return data.depth2[(np.abs(data - data.isel(depth2=1)) >
                                    criterion).argmax(axis=0)].drop('depth2')

        if self.kind == 'rama':
            temp = interp_to_1m(self.ctd['T'])
        else:
            temp = self.ctd['T'].bfill(dim='depth2')
        self.ild = find_mld(temp, 0.2)

        if self.kind == 'rama':
            rho = interp_to_1m(self.ctd.ρ)
            self.mld = find_mld(rho, 0.03)
            salt = interp_to_1m(self.ctd.S)
            self.sld = find_mld(salt, 0.02)
            self.bld = np.abs(self.mld - self.ild)
        else:
            self.mld = xr.zeros_like(self.ild) * np.nan
            self.ild = xr.zeros_like(self.ild) * np.nan
            self.sld = xr.zeros_like(self.ild) * np.nan
            self.bld = xr.zeros_like(self.ild) * np.nan

    def ReadSST(self, name='mur'):

        if name == 'mur':
            sst = xr.open_mfdataset('../datasets/mur/201*')
            sst = sst.analysed_sst
        else:
            raise ValueError('SST dataset ' + name + ' is not supported yet!')

        # read ±1° for gradients
        sst = sst.sel(lon=slice(self.lon - 1, self.lon + 1),
                      lat=slice(self.lat - 1, self.lat + 1)).load() - 273.15

        self.sst['T'] = sst.sel(lon=self.lon, lat=self.lat, method='nearest')

        sst.values = sp.ndimage.uniform_filter(sst.values, 10)
        zongrad = xr.DataArray(
            np.gradient(sst, sst.lon.diff(dim='lon').mean().values, axis=1),
            dims=sst.dims, coords=sst.coords, name='Zonal ∇T')

        mergrad = xr.DataArray(
            np.gradient(sst, sst.lat.diff(dim='lat').mean().values, axis=2),
            dims=sst.dims, coords=sst.coords, name='Meridional ∇T')

        zongrad /= 1e5
        mergrad /= 1e5

        self.sst = xr.Dataset()
        self.sst['Tx'] = zongrad.sel(lon=self.lon, lat=self.lat,
                                     method='nearest')
        self.sst['Ty'] = mergrad.sel(lon=self.lon, lat=self.lat,
                                     method='nearest')
        self.sst['time'] = self.sst.time.dt.floor('D')

    def ReadMet(self, fname: str=None, WindType='', FluxType=''):

        from dcpy.util import mdatenum2dt64

        if WindType == 'pmel':
            if fname is None:
                raise ValueError('I need a filename for PMEL met data!')

            met = xr.open_dataset(fname)
            spd = met.WS_401.where(met.WS_401 < 100).squeeze()
            wu = met.WU_422.where(met.WU_422 < 100).squeeze()
            wv = met.WV_423.where(met.WV_423 < 100).squeeze()
            z0 = abs(met['depu'][0])
            τ = xr.DataArray(airsea.windstress.stress(spd, z0, drag='smith'),
                             dims=['time'], coords=[met.time.values],
                             name='τ')
            tau = τ * np.exp(1j * np.angle(wu + 1j * wv))
            taux = xr.DataArray(tau.real, dims=['time'],
                                coords=[met.time.values],
                                name='taux')
            tauy = xr.DataArray(tau.imag, dims=['time'],
                                coords=[met.time.values],
                                name='tauy')

            self.met = xr.merge([self.met, τ, taux, tauy])

        elif FluxType == 'merged':
            from scipy.io import loadmat
            mat = loadmat(fname, squeeze_me=False)
            Jtime = mdatenum2dt64(mat['Jq']['t'][0][0][0] - 366)
            Jq0 = xr.DataArray(-mat['Jq']['nhf'][0][0][0],
                               dims=['Jtime'], coords=[Jtime],
                               name='Jq0')
            swr = xr.DataArray(-mat['Jq']['swf'][0][0][0],
                               dims=['Jtime'], coords=[Jtime],
                               name='swr')

            self.met = xr.merge([self.met, Jq0, swr])

        elif FluxType == 'pmel':
            # merged using StitchRamaFlux.py
            self.flux = xr.open_dataset('../rama/rama_flux/'
                                        + 'rama-' + str(self.lat)
                                        + 'n-fluxes.nc')

    def ReadNcep(self):
        ''' Read NCEP precip rate '''

        P = (xr.open_mfdataset('../ncep/prate*')
             .sel(lon=self.lon, lat=self.lat, method='nearest').load())

        P = P.rename(dict(time='Ptime', prate='P'))
        # convert from kg/m^2/s to mm/hr
        P *= 1 / 1000 * 1000 * 3600.0

        self.met = xr.merge([self.met, P])

    def ReadTropflux(self, loc):
        ''' Read tropflux data. Save in moor.tropflux'''

        swr = (xr.open_mfdataset(loc + '/swr_*.nc')
               .sel(latitude=self.lat, longitude=self.lon, method='nearest')
               .load())
        lwr = (xr.open_mfdataset(loc + '/lwr_*.nc')
               .sel(latitude=self.lat, longitude=self.lon, method='nearest')
               .load())
        tau = (xr.open_mfdataset(loc + '/tau_*.nc')
               .sel(latitude=self.lat, longitude=self.lon, method='nearest')
               .load())
        net = (xr.open_mfdataset(loc + '/netflux_*.nc')
               .sel(latitude=self.lat, longitude=self.lon, method='nearest')
               .load())

        taux = xr.open_mfdataset(loc + '/taux_*.nc').taux
        tauy = xr.open_mfdataset(loc + '/tauy_*.nc').tauy
        tx_y = taux.diff(dim='latitude')
        ty_x = tauy.diff(dim='longitude')

        # tropflux is 1° - lets use that to our advantage
        lat2m, _ = sw.dist([self.lat - 0.5, self.lat + 0.5],
                           self.lon, units='m')
        lon2m, _ = sw.dist(
            self.lat, [self.lon - 0.5, self.lon + 0.5], units='m')

        curl = ((ty_x / lon2m - tx_y / lat2m)
                .sel(latitude=self.lat, longitude=self.lon, method='nearest')
                .to_dataset(name='curl'))

        self.tropflux = xr.merge([self.tropflux, swr, lwr, net,
                                  tau, curl,
                                  taux.sel(latitude=self.lat,
                                           longitude=self.lon,
                                           method='nearest').load(),
                                  tauy.sel(latitude=self.lat,
                                           longitude=self.lon,
                                           method='nearest').load()])

    def calc_niw_input(self, debug=False):
        '''
            Band passes top-most velocity bin and calculates flux with local or
            TropFlux winds
        '''
        # 1/1.25, 1.25 taken from Alford (2003)
        freqs = np.array([1 / 1.25, 1.25]) * self.inertial.values

        def filt(comp, freqs=freqs):
            ufilt = xfilter.bandpass(np.real(comp), coord='time',
                                     freq=freqs, cycles_per='D', order=2)
            vfilt = xfilter.bandpass(np.imag(comp), coord='time',
                                     freq=freqs, cycles_per='D', order=2)

            return (ufilt + 1j * vfilt)

        if 'depth' not in self.vel.u.dims:
            self.vel = self.vel.expand_dims(['depth'])

        # choose ML velocity: pick topmost bin
        uinterp = self.vel.u.isel(depth=0).squeeze().interpolate_na('time')
        vinterp = self.vel.v.isel(depth=0).squeeze().interpolate_na('time')

        # reintroduce nans after filtering
        ZI = (filt(uinterp + 1j * vinterp)
              .where(~np.isnan(self.vel.u.isel(depth=0))))

        if debug:
            _, ax = dcpy.ts.PlotSpectrum(uinterp + 1j * vinterp, twoside=False)
            dcpy.ts.PlotSpectrum(ZI.dropna('time'), ax=ax, twoside=False)
            dcpy.plots.linex(freqs)

        # choose wind-stress
        if 'taux' in self.met:
            T = (self.met.taux + 1j * self.met.tauy)
        else:
            T = (self.tropflux.taux + 1j * self.tropflux.tauy)

        That = filt(T)
        if debug:
            dcpy.ts.PlotSpectrum(T.dropna('time'), ax=ax, twoside=False)
            dcpy.ts.PlotSpectrum(That.dropna('time'), ax=ax, twoside=False)

        That = 1 / 1025 * (That.interp_like(ZI).dropna('time'))

        # calculate flux
        self.niw['true_flux'] = np.real(1025 * np.conj(ZI) * That)
        self.niw['true_flux'].attrs['long_name'] = 'Local wind input'
        self.niw['true_flux'].attrs['units'] = 'W/m²'

    def AddEvents(self, name, t0, t1, pods=None):

        if pods is None:
            pods = list(self.χpod.keys())

        if type(pods) is not list:
            pods = [pods]

        for pp in pods:
            self.χpod[pp].events[name] = _decode_time(t0, t1)

        # append to the mooring list
        self.events[name] = self.χpod[pp].events[name]

    def AddSeason(self):

        # canonical values for 12N
        seasons = dict()
        # seasons[2014] = {
        #     'NE': ('2013-Dec-01', '2014-Feb-14'),
        #     'NESW':  ('2014-Feb-15', '2014-May-05'),
        #     'SW': ('2014-May-06', '2014-Sep-24'),
        #     'SWNE': ('2014-Sep-25', '2014-Dec-12')
        # }

        # seasons[2015] = {
        #     'NE': ('2014-Dec-12', '2015-Mar-01'),
        #     'NESW': ('2015-Mar-01', '2015-May-15'),
        #     'SW': ('2015-May-16', '2015-Oct-14'),
        #     'SWNE': ('2015-Oct-15', '2015-Dec-01')
        # }

        seasons[2014] = {
            'NE': ('2013-Dec-01', '2014-Mar-01'),
            'NESW': ('2014-Mar-01', '2014-May-01'),
            'SW': ('2014-May-01', '2014-Oct-01'),
            'SWNE': ('2014-Oct-01', '2014-Dec-01')
        }

        seasons[2015] = {
            'NE': ('2014-Dec-01', '2015-Mar-01'),
            'NESW': ('2015-Mar-01', '2015-Apr-30'),
            'SW': ('2015-Apr-30', '2015-Sep-30'),
            'SWNE': ('2015-Oct-1', '2015-Nov-30')
        }

        self.season = seasons
        # save in datetime format
        for year in seasons.keys():
            for name in seasons[year]:
                t0, t1 = seasons[year][name]
                self.season[year][name] = _decode_time(t0, t1)

    def AddDeployment(self, name, t0, t1):

        self.deployments.append(name)
        self.deploy[name] = _decode_time(t0, t1)

    def calc_near_inertial_input_complex(self):
        '''
            Uses complex demodulation. Not sure if this is a good idea
            since the inertial peak is broad.
        '''
        if self.kind !=  'rama':
            loc = '/home/deepak/datasets/ncep/'

            uwind = (xr.open_mfdataset(loc + 'uwnd*')
                     .sel(lon=self.lon, lat=self.lat, method='nearest')
                     .uwnd
                     .load())

            vwind = (xr.open_mfdataset(loc + 'vwnd*.nc')
                     .sel(lon=self.lon, lat=self.lat, method='nearest')
                     .vwnd
                     .load())

            tau = xr.DataArray(
                airsea.windstress.stress(np.hypot(uwind, vwind))
                * np.exp(1j * np.angle(uwind + 1j * vwind)),
                dims=uwind.dims, coords=uwind.coords)

        else:
            tau = (self.met.taux.interp(time=self.vel.time)
                   + 1j * self.met.tauy.interp(time=self.vel.time))

        # complex demodulate to get near-inertial currents
        dm = dcpy.ts.complex_demodulate(
            self.vel.w.isel(depth=0).squeeze(),
            central_period=1 / self.inertial / 1.05,
            cycles_per='D', bw=0.3, debug=False
        )

        niwflux = tau.real * dm.cw.real + tau.imag * dm.cw.imag

        self.niw = xr.Dataset()
        self.niw['u'] = dm.cw.real
        self.niw['v'] = dm.cw.imag
        self.niw['KE'] = self.niw.u**2 + self.niw.v**2
        self.niw['amp'] = np.abs(dm.cw)
        self.niw['pha'] = xr.DataArray(np.angle(dm.cw),
                                       dims=self.niw.amp.dims,
                                       coords=self.niw.amp.coords)
        self.niw['flux'] = niwflux

        plt.figure()
        np.abs(dm.cw).plot()
        np.abs(dm.ccw).plot()
        plt.legend(['cw', 'ccw'])

        f, ax = plt.subplots(4, 1, sharex=True)
        self.met.τ.plot(ax=ax[0])
        niwflux.plot(ax=ax[1])

        tau.real.plot(ax=ax[2])
        dm.cw.real.plot(ax=ax[2])
        tau.imag.plot(ax=ax[3])
        dm.cw.imag.plot(ax=ax[3])

        [self.MarkSeasonsAndEvents(aa) for aa in ax]

        return niwflux, dm

    def ReadVel(self, fname, FileType: str='ebob'):
        ''' Read velocity data '''

        if FileType == 'pmel':
            self.vel = xr.open_dataset(fname)

            self.vel = (self.vel.rename({'U_320': 'u',
                                         'V_321': 'v'})
                        .squeeze())
            self.vel.u.load()
            self.vel.v.load()

            # to m/s
            self.vel['u'] /= 100.0
            self.vel['v'] /= 100.0

            self.vel['u'].values[self.vel.u > 5] = np.nan
            self.vel['v'].values[self.vel.v > 5] = np.nan

        if FileType == 'ebob':
            from scipy.io import loadmat
            import dcpy.util

            adcp = loadmat('../ebob/ancillary/adcp/' + self.name + '.mat')

            z = adcp['depth_levels'].squeeze()
            time = dcpy.util.mdatenum2dt64(adcp['date_time'] - 366).squeeze()
            self.vel = xr.Dataset({'u': (['depth', 'time'], adcp['u'] / 100),
                                   'v': (['depth', 'time'], adcp['v'] / 100)},
                                  coords={'depth': z, 'time': time})

            if self.name == 'NRL2':
                # last few depth levels are nonsense it looks like.
                self.vel = self.vel.isel(depth=slice(None, -6))

            self.vel['uz'] = self.vel.u.differentiate('depth')
            self.vel['vz'] = self.vel.v.differentiate('depth')
            self.vel['shear'] = np.hypot(self.vel.uz, self.vel.vz)
            self.vel.time.values = self.vel.time.dt.round('H')
            self.vel.uz.attrs['units'] = '1/s'
            self.vel.vz.attrs['units'] = '1/s'

        self.vel['w'] = self.vel.u + 1j * self.vel.v

        self.vel.depth.attrs['units'] = 'm'
        self.vel.u.attrs['units'] = 'm/s'
        self.vel.v.attrs['units'] = 'm/s'

    def AddChipod(self, name, depth: int,
                  best: str, fname: str='Turb.mat', dir=None, avoid_wda=False):

        import chipy

        if dir is None:
            dir = self.datadir

        if avoid_wda and 'w' in best:
            best = best[:-1]

        self.χpod[name] = chipy.chipod(dir + '/data/',
                                       str(name), fname,
                                       best, depth=depth)

    def SetColorCycle(self, ax):

        z = np.array(list(self.zχpod.isel(time=1).values()))
        ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        corder = np.array(ccycle[:len(z)])

        for zz in z:
            cex = corder[z == zz]
            corder[z == zz] = cex[0]

        ccycle[:len(z)] = corder
        ax.set_prop_cycle('color', list(corder))

    def ChipodSeasonalSummary(self, pods=[], ax=None, filter_len=86400,
                              labels=[]):

        import matplotlib.pyplot as plt

        if ax is None:
            plt.figure(figsize=[7.5, 4.5])
            ax = plt.gca()
            ax.set_title(self.name)

        handles = []
        clabels = []
        pos = []

        if pods == []:
            pods = list(self.χpod.keys())

        if type(pods) is not list:
            pods = list(pods)

        for idx, name in enumerate(pods):
            hdl, lbl, p = self.χpod[name].SeasonalSummary(
                ax=ax, idx=idx, filter_len=filter_len)
            handles.append(hdl)
            clabels.append(lbl)
            pos.append(p)

        ax.set_title(self.name)

        if len(self.χpod) > 1:
            if labels == []:
                labels = clabels

            import numpy as np
            ax.set_xticks(list(np.mean(pos, 0)))
            ax.legend((handles[0]['medians'][0], handles[-1]['medians'][0]),
                      labels)

            limy = ax.get_yticks()
            limx = ax.get_xticks()
            ax.spines['left'].set_bounds(limy[1], limy[-2])
            ax.spines['bottom'].set_bounds(limx[0], limx[-1])

        if filter_len is not None:
            ax.set_title(ax.get_title() + ' | filter_len=' + str(filter_len) +
                         ' s')

        return handles, labels, pos

    def DepthPlot(self, varname, est: str='best', filter_len=86400):

        import matplotlib.cm as cm

        s0 = 4  # min size of circles
        ns = 3  # scaling for sizea
        alpha = 0.4

        plt.figure(figsize=(6.5, 8.5))
        ax0 = plt.gca()
        ndt = np.round(1 / 4 / (self.ctd.time[1] - self.ctd.time[0]))
        hpc = ax0.pcolormesh(
            self.ctd.time[::ndt],
            -self.ctd.depth,
            self.ctd.temp[::ndt, :].T,
            cmap=plt.get_cmap('RdYlBu_r'),
            zorder=-1)
        ax0.set_ylabel('depth')

        xlim = [1e10, -1e10]
        hscat = [1, 2]
        for idx, unit in enumerate(self.χpod):
            pod = self.χpod[unit]
            xlim[0] = np.min([xlim[0], pod.time[1]])
            xlim[1] = np.max([xlim[1], pod.time[-2]])

            var, titlestr, scale, _ = pod.ChooseVariable(varname, est)
            time, var = pod.FilterEstimate(
                varname, pod.time, var, filter_len=filter_len, subset=True)
            if scale == 'log':
                normvar = np.log10(var)
                dn = normvar - np.nanmin(normvar)
                size = s0 + ns * s0 * dn / np.nanstd(dn)

            # running average depths,
            # then interpolate to correct time grid
            time2, depth2 = pod.FilterEstimate('Jq', pod.ctd1.time,
                                               -pod.ctd1.z, filter_len, True)
            depth = np.interp(time, time2, depth2)

            hscat[idx] = ax0.scatter(
                time,
                depth,
                s=size,
                c=np.log10(var),
                alpha=alpha,
                cmap=cm.Greys)
            if idx == 0:
                clim = [np.nanmin(np.log10(var)), np.nanmax(np.log10(var))]
            else:
                clim2 = [np.nanmin(np.log10(var)), np.nanmax(np.log10(var))]
                clim = [min([clim[0], clim2[0]]), max([clim[1], clim2[1]])]

        hscat[0].set_clim(clim)
        hscat[1].set_clim(clim)
        plt.colorbar(hscat[0], ax=ax0, orientation='horizontal')
        plt.colorbar(hpc, ax=ax0, orientation='horizontal')
        ax0.xaxis_date()
        ax0.set_title(self.name + ' | ' + titlestr)
        ax0.set_xlim(xlim)

    def PlotFlux(self, ax, t, Jq, alpha=0.5, **kwargs):

        negcolor = '#79BEDB'
        poscolor = '#e53935'

        Jq1 = Jq.copy()
        Jq1[Jq1 > 0] = 0
        ax.fill_between(t, Jq1, linestyle='-',
                        color=negcolor, linewidth=1,
                        alpha=alpha, **kwargs)
        Jq1 = Jq.copy()
        Jq1[Jq1 < 0] = 0
        ax.fill_between(t, Jq1, linestyle='-',
                        color=poscolor, linewidth=1,
                        alpha=alpha, **kwargs)

        # for tt in ax.get_yaxis().get_ticklabels():
        #     if tt.get_position()[1] < 0:
        #         tt.set_color(negcolor)

        #     if tt.get_position()[1] > 0:
        #         tt.set_color(poscolor)

    def avgplt(self, ax, t, x, flen, filt, axis=-1, decimate=True, **kwargs):
        from dcpy.util import MovingAverage
        from dcpy.ts import BandPassButter

        x = x.copy()

        x_is_xarray = type(x) is xr.core.dataarray.DataArray
        if type(t) is xr.core.dataarray.DataArray:
            dt = (t[3] - t[2]).values
        else:
            dt = (t[3] - t[2]) * 86400

        if np.issubdtype(dt.dtype, np.timedelta64):
            dt = dt.astype('timedelta64[s]').astype('float32')

        if flen is not None:
            if filt == 'mean':
                if x_is_xarray:
                    N = np.int(np.floor(flen / dt))
                    a = x.rolling(time=N, center=True, min_periods=1).mean()
                    a = a.isel(time=slice(N - 1, len(a['time']) - N + 1, N))
                    t = a['time']
                    return t, a
                else:
                    t = MovingAverage(t.copy(), flen / dt,
                                      axis=axis, decimate=decimate)
                    x = MovingAverage(x.copy(), flen / dt,
                                      axis=axis, decimate=decimate)

            elif filt == 'bandpass':
                flen = np.array(flen.copy())
                assert(len(flen) > 1)
                x = BandPassButter(x.copy(), 1 / flen,
                                   dt, axis=axis, dim='time')

            else:
                from dcpy.util import smooth
                if x_is_xarray:
                    xnew = smooth(x.values.squeeze(), flen / dt)
                    x.values = np.reshape(xnew, x.shape)
                else:
                    x = smooth(x, flen / dt, axis=axis)

        if ax is None:
            return t, x
        else:
            hdl = ax.plot(t, x, **kwargs)
            ax.xaxis_date()
            return hdl[0]

    def Quiver(self, t=None, u=None, v=None, ax=None, flen=None,
               filt=None, color='k'):

        if ax is None:
            ax = plt.gca()

        if t is None:
            t = self.vel.time
            u = self.vel.u
            v = self.vel.v

        t1, u = self.avgplt(None, t, u, flen, filt)
        t1, v = self.avgplt(None, t, v, flen, filt)

        hquiv = ax.quiver(t1.values, np.zeros_like(t1.values),
                          u.values.squeeze(), v.values.squeeze(),
                          scale=3, color=color, alpha=0.4)
        ax.quiverkey(hquiv,
                     0.3, 0.2, 0.5, '0.5 m/s', labelpos='E',
                     coordinates='axes')

        return hquiv

    def MarkSeasonsAndEvents(self, ax=None, season=True, events=True,
                             zorder=-10):
        import matplotlib.dates as dt

        if ax is None:
            ax = plt.gca()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if season:
            # seasonColor = {
            #     'NE': '#fef3d3',  # 'beige',
            #     'NESW': '#E6F5DE',  # lemonchiffon
            #     'SW': '#E7F1FA',  # wheat
            #     'SWNE': 'white'  # honeydew
            # }
            seasonColor = {
                'NE': 'beige',  # 'beige',
                'NESW': 'lemonchiffon',  # lemonchiffon
                'SW': 'wheat',  # wheat
                'SWNE': 'honeydew'  # honeydew
            }

            for pp in self.season:
                for ss in self.season[pp]:
                    clr = seasonColor[ss]
                    xx = dt.date2num(self.season[pp][ss])
                    ax.fill_between(xx, 0, 1,
                                    transform=ax.get_xaxis_transform('grid'),
                                    facecolor=clr, alpha=0.9,
                                    zorder=zorder, edgecolor=None)

        if events:
            for ss in self.events:
                if isinstance(events, str):
                    if ss != events:
                        continue

                xx = dt.date2num(self.events[ss])
                ax.fill_between(xx, 0, 1,
                                transform=ax.get_xaxis_transform('grid'),
                                facecolor='w', alpha=0.7,
                                zorder=zorder + 1)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def PlotCTD(self, name, ax=None, filt=None, filter_len=None,
                add_mld=True, kind='timeseries', lw=1, region={}, **kwargs):

        import cmocean as cmo

        if ax is None:
            ax = plt.gca()

        if name == 'T':
            cmap = mpl.cm.YlOrBr
        else:
            cmap = cmo.cm.haline_r

        region = self.select_region(region)

        # filter before subsetting
        var = dcpy.ts.xfilter(self.ctd[name], dim='time',
                              kind=filt, flen=filter_len, decimate=True)
        var = var.sel(**region)

        label = '$' + self.ctd[name].name + '$'

        # ebob hacks
        if self.ctd.depth.ndim > 1 and name == 'S':
            if kind == 'timeseries':
                # compress depth dimension
                var['depth'] = self.ctd.S.depth.median(dim='time')
            else:
                kwargs['x'] = 'time'
                kwargs['y'] = 'depth'

        if self.kind == 'ebob' and name == 'T' and kind == 'timeseries':
            # too many depths to timeseries!
            kind = 'pcolor'

        if kind == 'timeseries':
            from cycler import cycler
            N = len(var.depth)
            colors = mpl.cm.Greys_r(np.arange(N + 1) / (N + 1))
            ax.set_prop_cycle(cycler('color', colors))

            # more ebob hackery
            if 'depth2' in var.dims:
                ydim = 'depth2'
            else:
                if 'z' in var.dims:
                    ydim = 'z'
                else:
                    ydim = 'depth'

            hdl = (var.isel({ydim: slice(0, -1)})
                   .plot.line(x='time', hue=ydim, ax=ax,
                              add_legend=False, lw=0.5))

            ncol = N if N < 5 else 5
            ax.legend([str(aa) + 'm'
                       for aa in np.int32(np.round(var.depth))],
                      ncol=ncol)
            ax.set_ylabel(label)

        # if kind == 'profiles':
        #     var -= np.nanmean(var, axis=0)
        #     var += tV
        #     dt = (tV[1, 0] - tV[0, 0]) * 86400.0
        #     if filter_len is not None:
        #         N = np.int(np.ceil(filter_len / dt)) * 2
        #     else:
        #         N = 500

        #     # doesn't work yet
        #     hdl = ax.plot(var.T[::N, :], zV.T[::N, :])

        if kind == 'pcolor' or kind == 'contourf':
            if 'robust' not in kwargs:
                kwargs['robust'] = True

            if 'yincrease' not in kwargs:
                kwargs['yincrease'] = False

            hdl = []
            if self.kind == 'ebob' and name == 'S':
                t = np.broadcast_to(var.time, var.depth.shape)
                hdl.append(ax.scatter(t, var.depth, s=5, c=var))
                ax.invert_yaxis()
            else:
                hdl.append(var.plot.contourf(ax=ax, levels=25,
                                             cmap=cmap, zorder=-1,
                                             add_colorbar=True,
                                             **kwargs))
                if filt == 'bandpass':
                    levels = 20
                    thick_contours = None
                else:
                    if name == 'T':
                        levels = np.arange(25, 30, 1)
                        thick_contours = [28]
                    elif name == 'S':
                        levels = np.arange(31, 36, 1)
                        thick_contours = [32]
                    elif name == 'ρ':
                        levels = 10
                        thick_contours = None

                hdl.append(var.plot.contour(ax=ax, levels=levels,
                                            colors='k',
                                            linewidths=0.25,
                                            zorder=-1, **kwargs))

                if thick_contours:
                    hdl.append(var.plot.contour(ax=ax, levels=thick_contours,
                                                colors='k',
                                                linewidths=1,
                                                zorder=-1, **kwargs))

                if self.kind == 'rama':
                    ax.set_yticks(self.ctd.depth)

                ax.set_ylabel('depth')

        if kind == 'contour':
            hdl = var.plot.contour(ax=ax, levels=20, colors='gray',
                                   linewidths=0.25, zorder=-1,
                                   **kwargs)
            ax.set_ylabel('depth')

        if kind == 'pcolor' or kind == 'contour' or kind == 'contourf':
            if add_mld:
                # ild = dcpy.ts.xfilter(self.ild, dim='time',
                #                       kind=filt, flen=filter_len,
                #                       decimate=True)
                # ild = ild.sel(**region)
                # ild.plot(ax=ax, lw=1, color='lightgray')

                mld = dcpy.ts.xfilter(self.mld, dim='time',
                                      kind=filt, flen=filter_len,
                                      decimate=True)
                mld = mld.sel(**region)
                mld.plot(ax=ax, lw=1, color='darkgray')
                # # self.mld.plot(ax=ax, lw=1, color='darkgray')

            _corner_label(label, ax=ax)
            if self.kind == 'ebob':
                self.PlotχpodDepth(ax=ax, color='k')
            else:
                self.PlotχpodDepth(ax=ax)

        return hdl

    def PlotχpodDepth(self, ax=None, color='lightgray', **kwargs):
        from dcpy.ts import xfilter

        if ax is None:
            ax = plt.gca()

        ax.plot(self.χ.time,
                self.χ.z.transpose().pipe(xfilter, kind='mean', flen=86400),
                color=color, **kwargs)

    def select_region(self, region):

        if isinstance(region, list) and len(region) == 1:
            region = region[0]

        if not isinstance(region, dict):
            name = region
            if isinstance(name, str):
                if name in self.events:
                    t0, t1 = self.events[name]

                if name in self.deploy:
                    t0, t1 = self.deploy[name]

                if name.upper() in ['SW', 'SWNE', 'NE', 'NESW']:
                    raise ValueError('region must include year'
                                     ' along with season')

            if isinstance(name, list):
                if name[0] in self.season:
                    t0, t1 = self.season[name[0]][name[1].upper()]
                else:
                    raise ValueError(
                        str(name[0]) + ' not in ' + self.name + '.season')

            region = {'time': slice(t0, t1)}

        return region

    def met_turb_summary(self, filt='mean', filter_len=86400, region={},
                         met='tropflux', naxes=2, ax=[]):

        from dcpy.ts import xfilter

        if len(ax) == 0:
            f, axx = plt.subplots(naxes, 1, sharex=True,
                                  constrained_layout=True,
                                  subplot_kw=dict(facecolor=(1, 1, 1, 0)))
        else:
            axx = ax

        ax = {'met': axx[0], 'Kt': axx[1]}
        if naxes > 2:
            ax['rest'] = axx[2:]

        region = self.select_region(region)

        filtargs = {'kind': filt, 'decimate': True,
                    'flen': filter_len, 'dim': 'time'}
        lineargs = {'x': 'time', 'hue': 'depth',
                    'linewidth': 1, 'add_legend': False}
        tauargs = {'color': 'k', 'linewidth': lineargs['linewidth'],
                   'zorder': 1, 'ax': ax['met']}

        # ------------ τ
        tau = 'tropflux'

        if 'time' in region:
            metregion = {'time': region['time']}
        else:
            metregion = {}

        if tau == 'tropflux':
            try:
                (self.tropflux.tau
                 .pipe(xfilter, **filtargs)
                 .sel(**metregion)
                 .plot(**tauargs))
            except ValueError:
                (self.tropflux.tau
                 .sel(**metregion)
                 .plot(**tauargs))
        else:
            (self.met.τ
             .pipe(xfilter, **filtargs)
             .sel(**metregion)
             .plot(**tauargs))

        if filt == 'bandpass':
            ax['met'].set_ylim([-0.1, 0.1])
        else:
            ax['met'].set_ylim([0, 0.3])

        ax['met'].set_ylabel('$τ$ (N/m²)')

        # ------------ flux
        ax['Jq0'] = ax['met'].twinx()
        ax['Jq0'].set_zorder(-1)

        if 'Jq0' not in self.met:
            # no mooring flux
            met = 'tropflux'

        if met == 'tropflux':
            Jq0 = self.tropflux.netflux
        elif 'Jq0' in self.met:
            Jq0 = self.met.Jq0

        try:
            Jq0 = xfilter(Jq0.rename({'Jtime': 'time'}), **filtargs)
        except ValueError:
            # filter length is not long enough, i.e. data is too coarse
            pass

        self.PlotFlux(ax['Jq0'], Jq0.sel(**metregion).time.values,
                      Jq0.sel(**metregion).values)
        ax['Jq0'].set_ylabel('netflux (W/m²)', labelpad=0)
        ax['Jq0'].spines['right'].set_visible(True)
        ax['Jq0'].spines['left'].set_visible(False)
        ax['Jq0'].xaxis_date()

        if filt == 'bandpass':
            ax['Jq0'].set_ylim(
                np.array([-1, 1]) * np.max(np.abs(ax['Jq0'].get_ylim())))

        hkt = (self.KT.copy()
               .pipe(xfilter, **filtargs)
               .sel(**region)
               .plot.line(ax=ax['Kt'], **lineargs))

        ax['Kt'].set_yscale('log')
        ax['Kt'].set_title('')
        ax['Kt'].set_ylabel('$K_T$ (m²/s)')
        ax['met'].set_title('')
        ax['Jq0'].set_title('')
        ax['Kt'].set_ylim([1e-7, 1e-1])

        self.MarkSeasonsAndEvents(ax['Jq0'], events=False)
        self.MarkSeasonsAndEvents(ax['Kt'], events=False)

        return ax, hkt

    def plot_niw_turb(self, region={}):

        region = self.select_region(region)

        f, ax = plt.subplots(3, 1, sharex=True)

        niw = self.niw.interp(
            depth=np.floor(self.zχpod.mean(dim='time').values))

        niw.KE.sel(**region).plot.line(x='time', hue='depth', ax=ax[0])
        (self.KT.sel(**region)
         .resample(time='6H').mean(dim='time')
         .plot.line(x='time', hue='depth', ax=ax[1]))
        (self.Jq.sel(**region)
         .resample(time='6H').mean(dim='time')
         .plot.line(x='time', hue='depth', ax=ax[2]))

        ax[1].set_yscale('log')
        [self.MarkSeasonsAndEvents(aa) for aa in ax]

    def Plotχpods(self, est: str='best', filt='mean', filter_len=86400,
                  quiv=False, TSkind='pcolor', region={},
                  Tlim=[None, None], Slim=[None, None], add_mld=False,
                  met='local', fluxvar='netflux', tau='local', event=None):
        ''' Summary plot for all χpods '''

        from dcpy.ts import xfilter

        lw = 0.5

        region = self.select_region(region)

        # initialize axes
        f = plt.figure(figsize=[11, 6], constrained_layout=True)
        f.set_constrained_layout_pads(h_pad=1/72.0)
        gs = f.add_gridspec(6, 2)

        ax = dict()
        ax['met'] = f.add_subplot(gs[0, 0])
        ax['met'].set_facecolor('none')
        ax['N2'] = f.add_subplot(gs[1, 0], sharex=ax['met'])

        ax['T'] = f.add_subplot(gs[0, 1], sharex=ax['met'])
        if TSkind !=  'timeseries':
            ax['S'] = f.add_subplot(gs[1, 1], sharex=ax['met'], sharey=ax['T'])
        else:
            ax['S'] = f.add_subplot(gs[1, 1], sharex=ax['met'])

        if self.vel:
            if TSkind !=  'timeseries' and self.kind == 'ebob':
                ax['u'] = f.add_subplot(gs[3, 0], sharex=ax['met'], sharey=ax['T'])

            if TSkind == 'timeseries' or self.kind != 'ebob':
                ax['u'] = f.add_subplot(gs[3, 0], sharex=ax['met'])

            if self.kind == 'ebob':
                ax['v'] = f.add_subplot(gs[4, 0], sharex=ax['met'], sharey=ax['u'])
                ax['shear'] = f.add_subplot(gs[2, 0],
                                            sharex=ax['met'], sharey=ax['u'])
            else:
                ax['v'] = ax['u']
                ax['χ'] = f.add_subplot(gs[4, 0], sharex=ax['met'])
                ax['shear'] = f.add_subplot(gs[2, 0], sharex=ax['met'])

        else:
            ax['χ'] = f.add_subplot(gs[3, 0], sharex=ax['met'])

        if self.ssh is not []:
            ax['ssh'] = f.add_subplot(gs[5, 0], sharex=ax['met'])

        ax['niw'] = f.add_subplot(gs[2, 1], sharex=ax['met'], sharey=ax['T'])
        ax['Tz'] = f.add_subplot(gs[3, 1], sharex=ax['met'])
        ax['Kt'] = f.add_subplot(gs[4, 1], sharex=ax['met'])
        ax['Jq'] = f.add_subplot(gs[5, 1], sharex=ax['met'])

        filtargs = {'kind': filt, 'decimate': True,
                    'flen': filter_len, 'dim': 'time'}
        lineargs = {'x': 'time', 'hue': 'depth',
                    'linewidth': lw, 'add_legend': False}

        if filter_len is None:
            filt = None

        if filter_len is not None:
            titlestr = (self.name + ' | '
                        + self.GetFilterLenLabel(filt, filter_len))
        else:
            titlestr = self.name

        if event is not None:
            t0, t1 = self.events[event]
            dt = np.timedelta64(5, 'D')
            region['time'] = slice(t0 - dt, t1 + dt)
            titlestr += '| Event = ' + event

        plt.suptitle(titlestr, y=1.03)

        # ------------ τ
        if 'τ' not in self.met:
            tau = 'tropflux'

        tauargs = {'color': 'k', 'linewidth': lw, 'zorder': 1,
                   'ax': ax['met']}
        if tau == 'tropflux':
            try:
                (self.tropflux.tau
                 .pipe(xfilter, **filtargs)
                 .sel(**region)
                 .plot(**tauargs))
            except ValueError:
                (self.tropflux.tau
                 .sel(**region)
                 .plot(**tauargs))
        else:
            (self.met.τ
             .pipe(xfilter, **filtargs)
             .sel(**region)
             .plot(**tauargs))

        if filt == 'bandpass':
            ax['met'].set_ylim([-0.1, 0.1])
        else:
            ax['met'].set_ylim([0, 0.3])

        # ----------- precip
        if 'P' in self.met:
            self.avgplt(ax['met'], self.met.P.Ptime.values,
                        self.met.P.values / 10,
                        flen=filter_len, filt=filt, color='slateblue',
                        linewidth=lw, zorder=-1)

        # ------------ EKE
        if self.ssh:
            (self.ssh.EKE
             .pipe(xfilter, **filtargs)
             .sel(**region)
             .plot(ax=ax['ssh']))
            (self.ssh.sla
             .pipe(xfilter, **filtargs)
             .sel(**region)
             .plot(ax=ax['ssh']))
            ax['ssh'].set_title('')
            ax['ssh'].legend(['EKE', 'SSHA'])
            ax['ssh'].axhline(0, zorder=-10, color='gray', ls='--')
            ax['ssh'].set_ylabel('')

        # ------------ flux
        ax['Jq0'] = ax['met'].twinx()
        ax['Jq0'].set_zorder(-1)

        if 'Jq0' not in self.flux:
            # no mooring flux, fall back to tropflux
            met = 'tropflux'

        if met == 'tropflux':
            Jq0 = self.tropflux.netflux
        elif 'Jq0' in self.flux:
            Jq0 = self.flux.Jq0

        try:
            Jq0 = xfilter(Jq0, **filtargs)
        except ValueError:
            print('Skipping netflux')
            # filter length is not long enough, i.e. data is too coarse
            pass

        self.PlotFlux(ax['Jq0'],
                      Jq0.sel(**region).time.values,
                      Jq0.sel(**region))
        ax['Jq0'].set_ylabel(fluxvar + ' [W/m²]', labelpad=0)
        ax['Jq0'].spines['right'].set_visible(True)
        ax['Jq0'].spines['left'].set_visible(False)

        if filt == 'bandpass':
            ax['Jq0'].set_ylim(
                np.array([-1, 1]) * np.max(np.abs(ax['Jq0'].get_ylim())))

        ((self.N2.copy()
          .pipe(xfilter, **filtargs)
          .sel(**region) / 1e-4)
         .plot.line(ax=ax['N2'], **lineargs))

        Tz = self.Tz.copy().pipe(xfilter, **filtargs).sel(**region)
        (Tz.plot.line(ax=ax['Tz'], **lineargs))

        # ---------- χpods
        if 'χ' in ax:
            (self.χ.copy()
             .pipe(xfilter, **filtargs)
             .sel(**region)
             .plot.line(ax=ax['χ'], **lineargs))
            ax['χ'].set_yscale('log')
            ax['χ'].set_title('')
            ax['χ'].set_ylabel('$χ$')

        (self.KT.copy()
         .pipe(xfilter, **filtargs)
         .sel(**region)
         .plot.line(ax=ax['Kt'], **lineargs))
        ax['Kt'].set_yscale('log')
        ax['Kt'].set_title('')
        ax['Kt'].set_ylabel('$K_T$')

        Jqt = (self.Jq.copy()
               .pipe(xfilter, **filtargs)
               .sel(**region))
        Jqt.plot.line(ax=ax['Jq'], **lineargs)
        ax['Jq'].set_ylim(dcpy.plots.robust_lim(np.ravel(Jqt)))
        ax['Jq'].set_title('')
        ax['Jq'].axhline(0, color='gray', zorder=-1, linewidth=0.5)
        ax['Jq'].set_ylabel('$J_q^t$')
        # ax['Jq'].grid(True, axis='y', linestyle='--', linewidth=0.5)

        # -------- T, S
        ctdargs = dict(filt=filt, filter_len=filter_len, kind=TSkind,
                       lw=0.5, region=region, add_mld=add_mld)
        ax['Splot'] = self.PlotCTD('S', ax['S'], vmin=Slim[0], vmax=Slim[1],
                                   **ctdargs)
        ax['Tplot'] = self.PlotCTD('T', ax['T'], vmin=Tlim[0], vmax=Tlim[1],
                                   **ctdargs)

        # -------- NIW
        if 'KE' in self.niw:
            self.niw.sel(**region).KE.plot(ax=ax['niw'], robust=True,
                                           add_colorbar=False, cmap=mpl.cm.Reds)
            self.PlotχpodDepth(ax=ax['niw'], color='k')
            _corner_label('$KE_{in}$', ax=ax['niw'], y=0.1)
            ax['niw'].set_ylim([250, 0])
            ax['niw'].set_title('')

        ax['met'].set_ylabel('$τ$ (N/m²)')

        ax['N2'].set_title('')
        ax['N2'].legend([str(zz) + ' m' for zz in self.N2.depth.values])
        ax['N2'].set_ylabel('$N²$ ($10^{-4}$)')
        limy = ax['N2'].get_ylim()
        if filt != 'bandpass':
            ax['N2'].set_ylim([0, limy[1]])

        ax['Tz'].set_title('')
        ax['Tz'].set_ylabel('$\\partial T/ \\partial z$')
        ax['Tz'].axhline(0, color='gray', zorder=-1, linewidth=0.5)
        if np.sum(Tz.values < 0)/Tz.size > 0.3:
            ax['Tz'].set_yscale('symlog', linthreshy=1e-3, linscaley=0.5)
        ax['Tz'].grid(True, axis='y', linestyle='--', linewidth=0.5)

        # ------------ velocity
        if 'v' in ax:
            uplt = (self.vel.u.copy().squeeze()
                    .pipe(xfilter, **filtargs)
                    .sel(**region))
            vplt = (self.vel.v.copy().squeeze()
                    .pipe(xfilter, **filtargs)
                    .sel(**region))

            if self.kind == 'ebob':
                quiv = False

            if quiv:
                ax['hquiv'] = self.Quiver(self.vel.time, uplt, vplt,
                                          ax['v'])
                ax['v'].set_title('')
                ax['v'].set_yticklabels([])
                ax['v'].set_ylabel('(u,v)')

            else:
                if self.kind == 'rama':
                    ax['Uplot'] = uplt.plot(ax=ax['u'], lw=0.5)
                    ax['Vplot'] = vplt.plot(ax=ax['v'], lw=0.5)
                    ax['spdplot'] = (np.hypot(uplt, vplt)
                                     .plot(ax=ax['v'], lw=0.5, color='gray',
                                           zorder=-10))
                    zint = uplt.depth.values.astype('int32')
                    ax['u'].legend(('$u_{' + str(zint) + '}$',
                                    '$v_{' + str(zint) + '}$'), ncol=2)
                    ax['u'].axhline(0, color='gray', lw=0.5)
                    ax['v'].set_ylabel('(m/s)')
                    ax['v'].set_title('')

                if self.kind == 'ebob':
                    vargs = dict(robust=True, yincrease=False, levels=50,
                                 add_colorbar=False, cmap='RdBu_r', center=0)

                    udict = xr.plot.utils._determine_cmap_params(uplt.values,
                                                                 robust=True)
                    vdict = xr.plot.utils._determine_cmap_params(vplt.values,
                                                                 robust=True)
                    mn = np.min([udict['vmin'], vdict['vmin']])
                    mx = np.max([udict['vmax'], vdict['vmax']])

                    vargs['vmin'] = -1 * np.max(np.abs([mn, mx]))
                    vargs['vmax'] = np.max(np.abs([mn, mx]))

                    ax['Uplot'] = uplt.plot.contourf(ax=ax['u'], **vargs)
                    ax['Vplot'] = vplt.plot.contourf(ax=ax['v'], **vargs)

                    labelargs = dict(x=0.05, y=0.15, alpha=0.05)
                    _corner_label('$u$', **labelargs, ax=ax['u'])
                    _corner_label('$v$', **labelargs, ax=ax['v'])
                    self.PlotχpodDepth(ax=ax['u'], color='k')
                    self.PlotχpodDepth(ax=ax['v'], color='k')

                    ax['u'].set_ylim(ax['T'].get_ylim())
                    ax['v'].set_ylim(ax['T'].get_ylim())

        if self.kind == 'ebob':
            shhdl = (self.vel.shear.sel(**region)
                     .plot.contourf(x='time', ax=ax['shear'],
                                    add_colorbar=False, yincrease=False,
                                    robust=True, cmap=mpl.cm.Reds))
            ax['shear'].set_ylabel('depth')
            ax['shear'].set_ylim(ax['T'].get_ylim())
            self.PlotχpodDepth(ax=ax['shear'], color='k')
            _corner_label('|$(u_z, v_z)$|', x=0.15, y=0.15, ax=ax['shear'])
        elif self.kind == 'rama':
            if self.pitot and 'shear' in self.pitot:
                shhdl = (self.pitot.shear
                         .pipe(xfilter, **filtargs)
                         .sel(**region)
                         .plot.line(x='time', ax=ax['shear'], lw=0.5,
                                    add_legend=False))
                ax['shear'].set_ylabel('Shear (1/s)')
                ax['shear'].set_title('')
                ax['shear'].axhline(0, ls='-', lw=0.5, color='gray')

        ax['met'].set_xlim([self.χ.sel(**region).time.min().values,
                            self.χ.sel(**region).time.max().values])

        for name in ['N2', 'T', 'S', 'v', 'χ', 'Kt',
                     'Jq', 'Tz', 'shear', 'ssh']:
            if name in ax:
                self.MarkSeasonsAndEvents(ax[name])
                if filt == 'bandpass':
                    if (name not in ['T', 'S'] or
                            (name in ['T', 'S'] and TSkind == 'timeseries')):
                        dcpy.plots.liney(0, ax=ax[name])

        self.MarkSeasonsAndEvents(ax['met'], season=False)

        hcbar = dict()
        if isinstance(ax['Tplot'], mpl.contour.QuadContourSet):
            hcbar['T'] = _colorbar(ax['Tplot'])

        if (isinstance(ax['Splot'], mpl.contour.QuadContourSet)):
            hcbar['S'] = _colorbar(ax['Splot'][0])
        if isinstance(ax['Splot'][0], mpl.collections.PathCollection):
            hcbar['S'] = _colorbar(ax['Splot'][0])

        # if ax['S'].get_ylim()[0] > 300:
        #     ax['S'].set_ylim([300, 0])

        if 'Uplot' in ax and 'Vplot' in ax and self.kind == 'ebob':
            hcbar['uv'] = _colorbar(ax['Uplot'], [ax['u'], ax['v']])

        if 'shear' in ax and self.kind == 'ebob':
            hcbar['shear'] = _colorbar(shhdl, ax=[ax['shear']])

        for aa in ax:
            if isinstance(ax[aa], mpl.axes.Axes):
                ax[aa].set_xlabel('')
                if ~ax[aa].is_last_row():
                    [tt.set_visible(False) for tt in ax[aa].get_xticklabels()]

        return ax

    def plot_shear_mixing(self):
        uzi = self.interp_shear('bins')

        filter_kwargs = dict(cycles_per='D', coord='time', order=3)

        zpod = (self.zχpod.sel(num=1, drop=True)
                .interp(time=self.vel.time).dropna('time'))

        full = xr.Dataset()
        full['shear'] = (uzi.uz.interpolate_na('time')
                         + 1j * uzi.vz.interpolate_na('time'))
        full['u'] = uzi.u
        full['v'] = uzi.v
        full['N2'] = ((self.turb.N2).isel(depth=1)
                      .interp(time=zpod.time))
        full['Tz'] = (self.turb.Tz.isel(depth=1)
                      .interp(time=zpod.time))
        full = full.interpolate_na('time')

        low = full.apply(xfilter.lowpass, freq=0.15, **filter_kwargs)
        loni = full.apply(xfilter.lowpass, freq=2.1, **filter_kwargs)
        high = full.apply(xfilter.bandpass, freq=[0.15, 4], **filter_kwargs)
        niw = full.apply(xfilter.bandpass, freq=[0.15, 2.1], **filter_kwargs)

        for ds in [full, low, high, niw]:
            ds['ke'] = 0.5 * (ds.u**2 + ds.v**2)
            ds['ke'].attrs['long_name'] = 'KE'
            ds['ke'].attrs['units'] = 'm²/s²'
            ds['dkedz'] = ds.u * np.real(ds.shear) + ds.v * np.imag(ds.shear)
            ds['dkedz'].dc.set_name_units('dKE/dz', 'm/s²')

        f, axx = plt.subplots(3, 1, sharex=True, constrained_layout=True)
        ax = dict(zip(['shear', 'N2', 'KT'], axx))

        # np.abs(uzi.uz + 1j*uzi.vz).plot(color='gray')
        # np.abs(high).plot(ax=ax[0])
        # rs = dict(time='W', loffset='3D')
        rs = dict(time=7 * 24, center=True)
        var = 'shear'
        ((np.abs(full[var])**2)
         .plot(color='k', alpha=0.2, lw=1, ax=ax['shear'], label='Full'))
        (np.abs(low[var]).rolling(**rs).reduce(dcpy.util.ms)
         .plot(color='k', ax=ax['shear'], lw=2, zorder=10,
               label='RMS LF (< 7 days)'))
        (np.abs(loni[var]).rolling(**rs).reduce(dcpy.util.ms)
         .plot(color='k', ax=ax['shear'], lw=2, zorder=4, ls='--',
               label='RMS LF (< $M_2$)'))
        ((np.abs(niw.shear)).rolling(**rs).reduce(dcpy.util.ms)
         .plot(color='g', lw=2, ax=ax['shear'], zorder=-1,
               label='RMS BP (7d - $M_2$)'))
        ax['shear'].legend(loc='upper right')
        ax['shear'].set_ylim([0, 0.0001])
        ax['shear'].set_ylabel('$S²$')

        ((low.N2)
         .plot(ax=ax['N2'], color='k', lw=2, label='low passed'))
        ((full.N2)
         .plot(ax=ax['N2'], color='k', alpha=0.2, zorder=-2))
        ax['N2'].legend()
        ax['N2'].set_ylabel('$N²$')

        ax['z'] = ax['N2'].twinx()
        (zpod.resample(time='D', loffset='12H').mean('time')
         .plot(ax=ax['z'], color='C0', lw=2))
        dcpy.plots.set_axes_color(ax['z'], color='C0', spine='right')
        ax['z'].set_ylim(np.flip(np.array(ax['z'].get_ylim()) + [-10, +10]))

        # ((high.N2).resample(time='D', loffset='12H').mean('time')
        #  .plot(ax=ax['N2']))
        # ax22 = ax[2].twinx()
        # (self.turb.Tz.isel(depth=1)
        #  .resample(time='D', loffset='-12H').mean('time')
        #  .plot(ax=ax22, color='k'))
        # dcpy.plots.set_axes_color(ax22, 'k', 'right')

        # var = 'KT'; limy=[1e-7, 1e-3]
        var = 'ε'; limy=[1e-12, 1e-6]
        KT = self.turb[var].isel(depth=1).resample(time='D', loffset='12H')
        (self.turb[var].isel(depth=1).plot.line(
            x='time', ax=ax['KT'], alpha=0.2, color='k', label='10 min'))
        (KT.mean('time')
         .plot.line(x='time', ax=ax['KT'], yscale='log', lw=2,
                    color='k', label='Daily mean'))
        (KT.median('time')
         .plot.line(x='time', ax=ax['KT'], yscale='log', lw=2,
                    color='C4', label='Daily median'))
        ax['KT'].legend()
        ax['KT'].set_ylim(limy)
        t = self.ε.isel(depth=1).dropna('time').time.values

        [aa.grid() for aa in axx]
        [aa.set_title('') for aa in axx]
        [aa.set_xlabel('') for aa in axx]
        [self.MarkSeasonsAndEvents(ax=aa) for aa in axx]
        [aa.set_xlim([t[0], t[-1]]) for aa in axx]

        axx[0].set_title(self.name)
        f.set_size_inches((15, 8))
        axx[-1].tick_params(axis='x', labelrotation=0)
        # axx[-1].set_xlim(['2013-12-15', '2015-03-01'])

        def _setup_xgrid(ax):
            ax.xaxis.set_minor_locator(mpl.dates.WeekdayLocator(interval=1))
            ax.grid(True, which='minor', axis='x')

        [_setup_xgrid(aa) for aa in axx]

    def PlotVel(self, ax=None, region={}, filt=None, filter_len=None):

        from dcpy.ts import xfilter

        region = self.select_region(region)
        filtargs = {'kind': filt, 'decimate': True,
                    'flen': filter_len, 'dim': 'time'}
        # lineargs = {'x': 'time', 'hue': 'depth', 'linewidth': lw,
        #             'add_legend': False}

        uplt = (self.vel.u.copy().squeeze()
                .pipe(xfilter, **filtargs)
                .sel(**region))
        vplt = (self.vel.v.copy().squeeze()
                .pipe(xfilter, **filtargs)
                .sel(**region))

        if ax is None:
            f, axx = plt.subplots(2, 1, sharex=True, sharey=True)
            ax = dict(u=axx[0], v=axx[1])

        if self.kind == 'rama':
            ax['Uplot'] = uplt.plot(ax=ax['u'], lw=0.5)
            ax['Vplot'] = vplt.plot(ax=ax['v'], lw=0.5)
            ax['spdplot'] = (np.hypot(uplt, vplt)
                             .plot(ax=ax['v'], lw=0.5, color='gray',
                                   zorder=-10))
            zint = uplt.depth.values.astype('int32')
            ax['u'].legend(('$u_{' + str(zint) + '}$',
                            '$v_{' + str(zint) + '}$'), ncol=2)
            ax['u'].axhline(0, color='gray', lw=0.5)
            ax['v'].set_ylabel('(m/s)')
            ax['v'].set_title('')

        if self.kind == 'ebob':
            vargs = dict(robust=True, yincrease=False, levels=50,
                         add_colorbar=False, cmap='RdBu_r', center=0)

            udict = xr.plot.utils._determine_cmap_params(uplt.values,
                                                         robust=True)
            vdict = xr.plot.utils._determine_cmap_params(vplt.values,
                                                         robust=True)
            mn = np.min([udict['vmin'], vdict['vmin']])
            mx = np.max([udict['vmax'], vdict['vmax']])

            vargs['vmin'] = -1 * np.max(np.abs([mn, mx]))
            vargs['vmax'] = np.max(np.abs([mn, mx]))

            ax['Uplot'] = uplt.plot.contourf(ax=ax['u'], **vargs)
            ax['Vplot'] = vplt.plot.contourf(ax=ax['v'], **vargs)

            labelargs = dict(x=0.05, y=0.15, alpha=0.05)
            _corner_label('u', **labelargs, ax=ax['u'])
            _corner_label('v', **labelargs, ax=ax['v'])
            self.PlotχpodDepth(ax=ax['u'], color='k')
            self.PlotχpodDepth(ax=ax['v'], color='k')

            hcbar = _colorbar(ax['Uplot'], [ax['u'], ax['v']])

        return ax, hcbar

    def plot_spectrogram(self):
        import sciviscolor as svc

        vel = (self.vel.sel(depth=slice(0, 120))
               .mean(dim='depth')
               .drop(['shear', 'depth']))
        KE = 1 / 2 * (vel.u**2 + vel.v**2)

        kwargs = dict(dim='time', window=30 * 24, shift=2 * 24,
                      multitaper=True)
        plot_kwargs = dict(levels=15, cmap=svc.cm.blue_orange_div,
                           add_colorbar=True, yscale='log')

        spec = dcpy.ts.Spectrogram(KE, **kwargs, dt=1 / 24)
        spec.freq.attrs['units'] = 'cpd'
        spec.name = 'PSD(depth avg KE)'

        tf = dcpy.ts.TidalAliases(1 / 24)
        f0 = dcpy.oceans.coriolis(self.lat)

        f, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)
        # self.KT.isel(depth=0).plot.line(x='time', ax=ax[0])
        # ylim = ax[0].get_ylim()
        # ax[0].set_yscale('log')
        # ax[0].set_ylim(ylim)

        np.log10(spec).plot.contourf(x='time', ax=ax[0], **plot_kwargs)
        dcpy.plots.liney([tf['M2'], tf['M2'] * 2,
                          1 / (2 * np.pi / (f0 * 86400))],
                         ax=ax[0], zorder=10, color='black')

        turb = np.log10(self.KT.isel(depth=0)
                        .resample(time='H')
                        .mean(dim='time')
                        .interpolate_na(dim='time', method='linear')
                        .dropna(dim='time'))

        turb[~np.isfinite(turb)] = 1e-12
        specturb = dcpy.ts.Spectrogram(turb, **kwargs, dt=1 / 24)
        specturb.name = 'PSD(log$_{10}$ ' + turb.name + ')'
        specturb.freq.attrs['units'] = 'cpd'
        np.log10(specturb).plot(x='time', ax=ax[1], **plot_kwargs,
                                robust=True)

        # spec = dcpy.ts.Spectrogram(self.ctd['T'].sel(depth2=100), **kwargs,
        #                            dt=10/24/60)
        # spec.freq.attrs['units'] = 'cpd'
        # spec.name = 'PSD(T)'
        # np.log10(spec).plot(x='time', ax=ax[1, 0], **plot_kwargs)

        [aa.set_xlabel('') for aa in ax[:-1]]

        ax[0].set_xlim(['01-Jan-2014', '31-Dec-2014'])

        f.suptitle(self.name)
        f.set_size_inches((5, 6))

    def PlotSpectrum(self, varname, est='best', filter_len=None,
                     nsmooth=5, SubsetLength=None, ticks=None,
                     ax=None, **kwargs):

        for idx, unit in enumerate(self.χpod):
            ax = self.χpod[unit].PlotSpectrum(varname, est=est,
                                              ax=ax,
                                              filter_len=filter_len,
                                              nsmooth=nsmooth,
                                              SubsetLength=SubsetLength,
                                              ticks=ticks, **kwargs)

        return ax

    def LagCorr(self, metvar='tau', met='local',
                freqs=None, filter_len=None, season='SW'):

        from dcpy.util import dt64_to_datenum
        import dcpy.plots

        f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        f.set_size_inches((8.5, 4.5))
        ax = np.asarray(ax)

        titlestr = 'Cross-correlation '

        if met != 'tropflux':
            tτ, τ = self.avgplt(None,
                                self.met.τtime, self.met.τ,
                                filt='mean', flen=filter_len)
        else:
            tτ, τ = self.avgplt(None,
                                dt64_to_datenum(self.tropflux[metvar]
                                                .time.values),
                                self.tropflux[metvar].values,
                                filt='mean', flen=filter_len)

        if freqs is not None:
            # band pass filter
            from dcpy.ts import BandPassButter
            τ = BandPassButter(τ, freqs=freqs, dt=(tτ[3] - tτ[2]) * 86400.0)
            titlestr += str(1 / freqs / 86400.0) + ' day bandpassed '

        for idx, unit in enumerate(self.χpod):
            pod = self.χpod[unit]
            est = pod.best

            tJ, Jqt = pod.ExtractSeason(pod.time, pod.Jq[est], season)
            if filter_len is not None:
                # do some averaging
                tJ, Jqt = self.avgplt(None,
                                      tJ, Jqt,
                                      filt='mean',
                                      flen=filter_len)

            # sign is an indication of stratification
            # Jqt = np.abs(Jqt)

            if freqs is not None:
                # band pass filter
                from dcpy.ts import BandPassButter
                Jqt = BandPassButter(Jqt, freqs=freqs,
                                     dt=(tJ[3] - tJ[2]) * 86400.0)

            τi = np.interp(tJ, tτ, τ)

            if np.all(np.isnan(Jqt)):
                raise ValueError('Jqt is all NaN. Did you filter too much?')

            # calculate and plot cross-correlations
            plt.sca(ax.ravel()[idx])
            plt.xcorr(τi[~np.isnan(Jqt)],
                      Jqt[~np.isnan(Jqt)], maxlags=None)
            plt.title(pod.name)

        dcpy.plots.linex(0, ax=list(ax.ravel()))
        plt.suptitle(titlestr + metvar
                     + ' and $J_q^t$ | season=' + season, y=1.01)
        f.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top='off',
                        bottom='off', left='off', right='off')
        plt.grid(False)
        plt.ylabel("Correlation coeff.")
        plt.xlabel(metvar + " lag (days)")

        plt.tight_layout()

    def PlotAllSpectra(self, filter_len=None, nsmooth=5,
                       SubsetLength=None, ticks=None, **kwargs):

        plt.figure(figsize=(6.5, 8.5))

        ax1 = plt.subplot(411)
        self.PlotSpectrum('T', filter_len=filter_len,
                          nsmooth=nsmooth,
                          SubsetLength=SubsetLength,
                          ticks=ticks, ax=ax1, **kwargs)
        ax1.set_title(self.name)

        ax2 = plt.subplot(412)
        self.PlotSpectrum('χ', filter_len=filter_len,
                          nsmooth=nsmooth,
                          SubsetLength=SubsetLength,
                          ticks=ticks, ax=ax2, **kwargs)

        ax3 = plt.subplot(413)
        self.PlotSpectrum('KT', filter_len=filter_len,
                          nsmooth=nsmooth,
                          SubsetLength=SubsetLength,
                          ticks=ticks, ax=ax3, **kwargs)

        ax4 = plt.subplot(414)
        self.PlotSpectrum('Jq', filter_len=filter_len,
                          nsmooth=nsmooth,
                          SubsetLength=SubsetLength,
                          ticks=ticks, ax=ax4, **kwargs)

        for ax in [ax2, ax3, ax4]:
            ax.get_legend().set_visible(False)

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('')
            ax.set_xticklabels([])

    def TSPlot(self, pref=0, ax=None, varname='KT', varmin=1e-3,
               filter_len=86400):
        '''
        Create a TS plot using all mooring data.
        Highlight $varname > $varmin at CTDs above/below
        χpod depths.

        Input:
            pref : reference pressure level
            ax : axes handles
            varname : variable to highlight
            varmin : threshold, when exceeded $varname is marked.
            filter_len : averaging for $varname
        '''

        f, axes = plt.subplots(2, 2, constrained_layout=True,
                               sharex=True, sharey=True)
        ax = dict()
        ax['NE'] = axes[0, 0]
        ax['NESW'] = axes[0, 1]
        ax['SW'] = axes[1, 0]
        ax['SWNE'] = axes[1, 1]

        ax['NE'].set_xlim([31, 35.8])
        ax['NE'].set_ylim([16, 32])

        mask = getattr(self, varname) > varmin

        ctd = self.ctd.resample(time='H').mean('time')
        for season in ['NE', 'NESW', 'SW', 'SWNE']:
            mask = ctd.time.monsoon.labels == season
            T = ctd.T_S if 'T_S' in ctd else ctd['T']
            dcpy.oceans.TSplot(ctd.S.where(mask),
                               T.where(mask),
                               ax=ax[season],
                               plot_distrib=False,
                               hexbin=False,
                               size=2)
            ax[season].text(0.05, 0.9, season, transform=ax[season].transAxes)

        [aa.set_xlabel('') for aa in axes[0, :]]
        [aa.set_ylabel('') for aa in axes[:, 1]]

        f.suptitle(self.name)

        plt.savefig('images/ts-' + self.short_name + '.png')

    def PlotCoherence(self, ax, v1, v2, nsmooth=5, multitaper=True):

        if multitaper:
            f, Cxy, phase, siglevel = dcpy.ts.MultiTaperCoherence(
                v1, v2, dt=1, tbp=nsmooth)
            siglevel = siglevel[0]
        else:
            f, Cxy, phase, siglevel = dcpy.ts.Coherence(
                v1, v2, dt=1, nsmooth=nsmooth)

        hcoh = ax[0].plot(f, Cxy)
        dcpy.plots.liney(siglevel, ax=ax[0], color=hcoh[0].get_color())
        ax[1].plot(f, phase)

        ax[0].set_ylabel('Coh. amp.')
        ax[0].set_ylim([0, 1])
        ax[0].set_xlim([0, max(f)])

        ax[1].set_ylabel('Coh. Phase (degrees)')
        ax[1].set_xlabel('Period (days)')
        ax[1].set_ylim([-180, 180])
        ax[1].set_yticks([-180, -90, -45, 0, 45, 90, 180])
        ax[1].grid(True, axis='y')
        ax[1].set_xticklabels([''] + [
            str('{0:.1f}').format(1 / aa) for aa in ax[1].get_xticks()[1:]
        ])

    def ExtractTimeRange(self, t1, v1, t2, v2, ndayavg=None, season=None):
        ''' v1 is sampled to time range of v2 '''

        from dcpy.util import MovingAverage
        from dcpy.util import find_approx

        t2 = t2[~np.isnan(v2)].copy()
        v2 = v2[~np.isnan(v2)]

        # extract appropriate time range from met variable
        it0 = find_approx(t1, t2[0])
        it1 = find_approx(t1, t2[-1])
        v1 = v1[it0:it1]
        t1 = t1[it0:it1]

        # daily (ndayavg) average before taking spectrum
        if ndayavg is not None:
            dt1 = (t1[2] - t1[1]) * 86400
            v1 = MovingAverage(v1, ndayavg * 86400 / dt1)
            t1 = MovingAverage(t1, ndayavg * 86400 / dt1)

            dt2 = (t2[2] - t2[1]) * 86400
            v2 = MovingAverage(v2, ndayavg * 86400 / dt2)
            t2 = MovingAverage(t2, ndayavg * 86400 / dt2)

        # extract season if asked to
        if season is not None:
            from dcpy.util import ExtractSeason
            t1, v1 = ExtractSeason(t1, v1, season)
            t2, v2 = ExtractSeason(t2, v2, season)

        v2i = np.interp(t1, t2, v2)

        return t1, v1, v2i

    def GetFilterLenLabel(self, filt=None, filter_len=None):

        if filt is None:
            txt = 'unfiltered'
        else:
            if np.any(filter_len > 86399):
                txt = str(filter_len / 86400.0) + ' day'
            else:
                if np.any(filter_len > 3599):
                    txt = str(filter_len / 60.0 / 60.0) + ' hr'
                else:
                    txt = str(filter_len / 60.0) + ' min'

            txt += ' ' + filt

        return txt

    def PlotMetCoherence(self, metvars=['Jq', 'wind'], ndayavg=1, nsmooth=4,
                         fbands=None, season=None, multitaper=False,
                         filt=None, filter_len=None):

        if len(metvars) == 1:
            plt.figure(figsize=(10, 5.5))
            ax0 = plt.subplot(221)
            ax1 = plt.subplot(223)
            ax2 = plt.subplot(222)
            ax3 = plt.subplot(224, sharex=ax2)

        else:
            plt.figure(figsize=(8, 8))
            # ax0 = plt.subplot2grid([4, 2], [0, 0], colspan=2)
            # ax1 = plt.subplot2grid([4, 2], [1, 0], colspan=2)
            ax0 = plt.subplot(321)
            ax1 = plt.subplot(322)
            ax2 = plt.subplot(323)
            ax3 = plt.subplot(325, sharex=ax2)
            ax4 = plt.subplot(324)
            ax5 = plt.subplot(326, sharex=ax4)

        title = (self.name + ' | '
                 + self.GetFilterLenLabel(filt, filter_len))
        ax0.set_title(title)

        for metidx, metvar in enumerate(metvars):
            if metvar == '':
                continue

            if metvar == 'Jq':
                met = self.met.Jq0
                tmet = self.met.Jtime
                axes = [ax2, ax3]
                label = '$J_q^0$'
                t, m = self.avgplt(None, tmet, met, filter_len, filt)
                ax0.plot(t, m, 'k', label=label)
                self.PlotFlux(ax0, t, m, alpha=0.1)
                dcpy.ts.PlotSpectrum(met, ax=ax1, color='k')

            if metvar == 'wind':
                from dcpy.util import dt64_to_datenum
                met = self.tropflux.tau.values
                tmet = dt64_to_datenum(self.tropflux.tau.time.values)
                axes = [ax4, ax5]
                label = '$1000τ$'
                self.avgplt(
                    ax0, tmet, met * 1000, filter_len, filt, label=label)
                dcpy.ts.PlotSpectrum(10 * met, nsmooth=20, ax=ax1)

            for idx, unit in enumerate(self.χpod):
                pod = self.χpod[unit]

                v2 = pod.Jq[pod.best].copy()
                t1, v1, v2i = self.ExtractTimeRange(tmet.copy(),
                                                    met.copy(),
                                                    pod.time.copy(), v2,
                                                    ndayavg, season)
                # self.avgplt(ax0, pod.time, pod.chi['mm1']['dTdz']*1e3,
                #             filter_len, filt, label='$T_z (10^{-3})$',
                #             color=(0, 0.6, 0.5))

                self.PlotCoherence(axes, v1, v2i, nsmooth, multitaper)

                if metidx == 0:
                    xlim = ax0.get_xlim()
                    self.avgplt(ax0, t1, v2i, filter_len, filt,
                                label='$J_q^t$' + pod.name[5:])
                    ax0.set_xlim([max(xlim[0], t1[0]),
                                  min(xlim[1], t1[-1])])

                    dcpy.ts.PlotSpectrum(v2, dt=ndayavg,
                                         ax=ax1, nsmooth=nsmooth)
                else:
                    axes[0].set_ylabel('')
                    axes[1].set_ylabel('')

            axes[0].set_title('between ' + label + ' and $J_q^t$')
            if filter_len is not None:
                dcpy.plots.FillRectangle(86400 / filter_len, ax=axes[0])
                dcpy.plots.FillRectangle(86400 / filter_len, ax=axes[1])
            if fbands is not None:
                dcpy.plots.linex(fbands, ax=axes)

        ax0.legend(ncol=2)
        dcpy.plots.liney(0, ax0, linestyle='-')
        ax0.set_xticks(ax0.get_xticks()[::2])

        ax1.set_ylabel('PSD')
        if filter_len is not None:
            dcpy.plots.FillRectangle(86400 / filter_len, ax=ax1)
        if fbands is not None:
            dcpy.plots.linex(fbands, ax=ax1)

        plt.tight_layout()

        if len(metvars) == 1:
            return [ax0, ax1, ax2, ax3]
        else:
            return [ax0, ax1, ax2, ax3, ax4, ax5]

    def eps_jb0(self):

        ρ0 = 1025
        cp = 4200
        g = 9.81
        α = 1.7e-4

        jb0 = - g * α / ρ0 / cp * self.flux.Jq0
        Lmo = self.monin_obukhov()

        depths = [15, 30]

        f, ax = plt.subplots(4, 1, sharex=True)
        self.PlotFlux(ax[0], self.flux.Jq0.time.values, self.flux.Jq0.values)
        self.Jq.sel(depth=15).plot(ax=ax[0])
        # self.flux.Jq0.plot(ax=ax[0], color='k')
        dcpy.plots.liney(0, ax=ax[0])

        (jb0.where(self.flux.Jq0 < 0)
         .plot.line(x='time', color='k', add_legend=False, ax=ax[1],
                    label='$J_b^0$'))
        # ax[1].legend()
        hdl = ((self.ε.where(self.ε > 0).sel(depth=depths))
               .plot.line(x='time', ax=ax[1]))

        for iu, unit in enumerate(self.χpod):
            pod = self.χpod[unit]
            if 'w' in pod.best:
                eps2 = pod.chi[pod.best[:-1]]['eps']
                eps2[eps2 == 0] = np.nan
                ax[1].plot(pod.time, eps2, color=hdl[iu].get_color(), ls='--')

        ax[1].grid(True)
        ax[1].set_yscale('log')
        ax[1].set_xlim('2013-12-01', '2014-12-31')

        (-self.ε.mld).plot(ax=ax[2], color='dimgray')
        (-Lmo.where(Lmo > 0)).plot(ax=ax[2], color='lightgray')
        ax[2].legend(['MLD', 'Monin-Obukhov'])
        ax[2].set_ylim([-50, 0])

        for iz, zz in enumerate(depths):
            dcpy.plots.liney(-1 * zz, ax=ax[2],
                             color=hdl[iz].get_color(), zorder=10)

        axN2 = ax[3]
        self.N2.plot.line(x='time', ax=axN2)
        axN2.set_title('')
        axN2.set_yscale('log')

        plt.tight_layout()

        # # interpolate jb0 to ε times
        # jb0new = np.interp(self.ε.time.astype('float32'),
        #                    self.flux.time.astype('float32'),
        #                    jb0, left=np.nan, right=np.nan)

        # Lmonew =  np.interp(self.ε.time.astype('float32'),
        #                     self.flux.time.astype('float32'),
        #                     jb0, left=np.nan, right=np.nan)

        # mask = np.logical_and(np.logical_and(jb0new > 0,
        #                                      self.ε.mld <=40),
        #                       Lmonew <=40)
        # plt.figure()
        # # plt.loglog(jb0new[mask], self.ε.sel(depth=15)[mask], '.')
        # plt.loglog(jb0new[mask], self.ε.sel(depth=30)[mask], '.')

    def Budget(self, do_mld=False):

        ρ = 1025
        cp = 4000

        resample_args = {'time': '1H'}

        def average(x):
            return x.resample(time='3H').mean(dim='time')

        def ddt(x):
            dt = x.time.diff(dim='time') / np.timedelta64(1, 's')
            return x.diff(dim='time') / dt

        def interp_b_to_a(a, b):
            return np.interp(a.time.astype(np.float32),
                             b.time.astype(np.float32),
                             b.squeeze(),
                             left=np.nan, right=np.nan)

        # T = dcpy.ts.xfilter(self.ctd['T'], kind='hann', flen=3*3600.0)
        T = self.ctd['T'].resample(**resample_args).mean(dim='time')

        if do_mld:
            rho = self.ctd.ρ.resample(**resample_args).mean(dim='time')

            # mixed layer depth
            mld = rho.depth[(np.abs(rho - rho.isel(depth=1)) >
                             0.01 / 2).argmax(axis=0)].drop('depth')

        # interpolate temp to 1m grid, then calculate heat content
        timevec = T.time.values.astype('float32')
        T = T.interpolate_na('time')
        f = sp.interpolate.RectBivariateSpline(timevec,
                                               T.depth, T.values.T,
                                               kx=1, ky=1)
        idepths = np.arange(1, 100, 1)
        Ti = xr.DataArray(f(timevec, idepths),
                          dims=['time', 'depth'],
                          coords=[T.time, idepths])
        Q = ρ * cp * xr.apply_ufunc(sp.integrate.cumtrapz,
                                    Ti, Ti.depth,
                                    kwargs={'axis': -1})
        Q['depth'] = Q.depth[:-1]

        if do_mld:
            Qmld = Q.sel(time=mld.time, depth=mld, method='nearest')
            Tmld = Qmld / ρ / cp / mld
            dTdt = ddt(Tmld)

        # budget is dQ/dt = Jq0 - Ih + Jqt
        # where both Jqs are positive when they heat the surface
        # i.e. Jqt > 0 → heat is moving upward
        Jq0 = xr.DataArray(interp_b_to_a(Q, self.flux.Jq0),
                           dims=['time'], coords=[Q.time])

        swr = xr.DataArray(interp_b_to_a(Q, self.flux.swr),
                           dims=['time'], coords=[Q.time])

        # penetrative heating
        Ih = 0.45 * swr * np.exp(-(1 / 15) * Q.depth)

        Jqt = self.Jq
        Jqt = Jqt.resample(**resample_args).mean(dim='time')

        sfcflx = (Jq0 - Ih).resample(**resample_args).mean(dim='time')
        dJdz = ((sfcflx + Jqt) / Jqt.depth).where(Jqt.time == Jq0.time)
        dJdz.name = 'dJdz'

        dQdt = Q.pipe(average).pipe(ddt)

        velmean = self.vel.resample(time='D').mean(dim='time').squeeze()

        f, ax = plt.subplots(len(Jqt.depth[:-1]), 1, sharex=True)
        for iz, z in enumerate(Jqt.depth[:-1]):
            Qadv = (self.sst.Tx * velmean.u +
                    self.sst.Ty * velmean.v) * z * ρ * cp

            ((dQdt).sel(depth=z)
             .plot.line(x='time', color='k', label='dQ/dt', ax=ax[iz]))
            ((sfcflx).sel(depth=z)
             .pipe(average)
             .plot(x='time', label='absorbed heating', ax=ax[iz]))
            (Qadv.plot.line(x='time', ax=[iz], label='advection'))
            (Jqt.sel(depth=z)
             .plot.line(x='time', ax=ax[iz]))
            dcpy.plots.liney(0, ax=ax[iz])
            ax[iz].set_xlim('2013-12-01', '2014-12-31')
            ax[iz].set_ylim([-1000, 1000])

        ax[0].legend()
        ax[0].set_xlim('2013-12-01', '2014-11-30')

        # ((dQdt.sel(depth=z)-sfcflx)
        #  .pipe(average)
        #  .plot(x='time', label='Net heating', ax=ax[1]))

        # (np.hypot(self.vel.u,
        #           self.vel.v)
        #  .resample(time='W').mean(dim='time')
        #  .plot(ax=ax[1]))
        # ax[0].set_ylim([-150, 150])

        return ax

        # return xr.merge([dJdz, Qt, Jqt])

        # def plot(jq, ax):
        #     [jq.sel(time='2014-02-01', depth=z).plot(ax=axx, lw=0.5)
        #      for (axx, z) in zip(ax, jq.depth)]

        # f, ax = plt.subplots(3, 1)
        # plot(Jqt, ax)
        # plot(Jqt_1D, ax)

        # ax1 = plt.subplot(311)
        # plt.plot(MovingAverage(tT, 144), MovingAverage(T, 144))
        # ax1.xaxis_date()

        # plt.ylabel('T (1 day avg)')
        # plt.subplot(312, sharex=ax1)
        # a = lowess(dTdt, tavg, frac=0.025)
        # plt.plot(a[:, 0], a[:, 1] * 86400)
        # plt.ylabel('∂T/∂t (C/day)')
        # dcpy.plots.liney(0)
        # dcpy.plots.symyaxis()

        # plt.subplot(313, sharex=ax1)
        # a = lowess(Jqt.T, tavg, frac=0.025)
        # plt.plot(a[:, 0], a[:, 1])
        # plt.plot(tavg, Jq0)
        # a = lowess(Q, tavg, frac=0.025)
        # plt.plot(a[:, 0], a[:, 1])
        # plt.legend(['$J_q^t$', '$J_q^0$', '$Q_{avg}$'])
        # dcpy.plots.liney(0)

        # dcpy.plots.symyaxis()

        # plt.gcf().autofmt_xdate()

    def Summarize(self, savefig=False, **kwargs):

        if self.kind == 'ebob':
            TSkind = 'pcolor'
        else:
            TSkind = 'pcolor'

        name = self.name.replace(' ', '-')

        if name == 'RAMA-12N':
            self.Plotχpods(region={'time': '2014'}, **kwargs)
            if savefig:
                plt.savefig('images/summary-'
                            + name + '-2014.png', bbox_inches='tight')

            self.Plotχpods(region={'time': '2015'}, **kwargs)
            if savefig:
                plt.savefig('images/summary-'
                            + name + '-2015.png', bbox_inches='tight')

        else:
            self.Plotχpods(TSkind=TSkind, **kwargs)
            if savefig:
                plt.savefig('images/summary-'
                            + name + '.png', dpi=180, bbox_inches='tight')

    def plot_turb_wavelet(self):

        from dcpy.ts import wavelet

        tides = dcpy.ts.TidalAliases(1 / 24)

        def common(axes):
            for ax in axes.flat:
                ax.set_yscale('log')
                ax.invert_yaxis()
                ax.axhline(1 / tides['M2'], color='k')
                ax.axhline(1 / self.inertial, color='k')

        # get a range of depths that the χpods cover
        # calcualte mean KE over that depth & then wavelet transform
        meanz = self.zχpod.mean(dim='time')
        stdz = self.zχpod.std(dim='time')

        if self.kind == 'ebob':
            depth_range = slice((meanz.min() - 2 * stdz.max()).values,
                                (meanz.max() + 2 * stdz.max()).values)
        elif self.kind == 'rama':
            depth_range = self.vel.u.depth

        ke = wavelet((self.vel.u**2 + self.vel.v**2)
                     .sel(depth=depth_range)
                     .mean(dim='depth')
                     .squeeze()
                     .dropna(dim='time'),
                     dt=(self.vel.u.time.diff(dim='time').values[0]
                         / np.timedelta64(1, 'D')))
        ke.period.attrs['units'] = 'days'
        ke.power.attrs['long_name'] = 'KE power'

        if self.kind == 'ebob':
            shear = wavelet((self.vel.shear)
                            .sel(depth=depth_range)
                            .mean(dim='depth'),
                            dt=1 / 24)
            shear.period.attrs['units'] = 'days'
            shear.power.attrs['long_name'] = 'shear power'

        # T = wavelet((self.ctd.T_S[2, :]), dt=1/6/24)
        # T.period.attrs['units'] = 'days'
        # T.power.attrs['long_name'] = 'Temp. Power'

        flux = wavelet(self.tropflux.swr.sel(time=slice('2012', '2016')),
                       dt=1)
        flux.period.attrs['units'] = 'days'
        flux.power.attrs['long_name'] = 'Shortwave power'

        tau = wavelet(self.tropflux.tau.sel(time=slice('2013', '2015')),
                      dt=1)
        tau.period.attrs['units'] = 'days'
        tau.power.attrs['long_name'] = 'τ power'

        kwargs = dict(levels=15, robust=True,
                      center=False)

        ax = dict()
        f, axes = plt.subplots(3, 2, sharex=True, constrained_layout=True)
        f.set_size_inches((12, 6))

        ax['flux'] = axes[0, 0]
        ax['stress'] = axes[0, 1]

        ax['shear'] = axes[1, 0]
        ax['KE'] = axes[1, 1]

        ax['eps'] = axes[2, 0]
        ax['KT'] = axes[2, 1]

        (np.log10(ke.power)
         .plot.contourf(**kwargs, ax=ax['KE'],
                        cbar_kwargs=dict(label=('log$_{10}$' +
                                                ke.power.attrs['long_name']))))

        if self.kind == 'ebob':
            (np.log10(shear.power)
             .plot.contourf(**kwargs, ax=ax['shear'],
                            cbar_kwargs=dict(
                                label=('log$_{10}$ ' +
                                       shear.power.attrs['long_name']))))
        elif self.kind == 'rama':
            (np.log10(ke.power)
             .plot.contourf(**kwargs, ax=ax['shear'], cbar_kwargs=dict(
                 label=('log$_{10}$' + ke.power.attrs['long_name']))))

        (np.log10(flux.power)
         .plot.contourf(**kwargs, ax=ax['flux'], cbar_kwargs=dict(
             label=('log$_{10}$ ' + flux.power.attrs['long_name']))))
        (np.log10(tau.power)
         .plot.contourf(**kwargs, ax=ax['stress'], cbar_kwargs=dict(
             label=('log$_{10}$ ' + tau.power.attrs['long_name']))))

        lineargs = dict(x='time', yscale='log', lw=0.5)
        (self.ε.resample(time='D').mean(dim='time')
         .plot.line(ax=ax['eps'], **lineargs))

        (self.KT.resample(time='D').mean(dim='time')
         .plot.line(ax=ax['KT'], **lineargs))

        [aa.set_xlabel('') for aa in axes[:-1, :].flat]
        # ax[-1].set_xlim([self.KT.time.min().values,
        #                  self.KT.time.max().values])
        f.suptitle(self.name)

        common(axes[:-1, :])

        for aa in axes.flat:
            self.MarkSeasonsAndEvents(aa)

    def sample_along_chipod(self, dataset, chipod_num=-1, debug=False):
        if chipod_num > 1:
            raise ValueError('chipod_num must be either 0 or 1')

        zpod = (self.zχpod.isel(num=chipod_num, drop=True)
                .interp(time=dataset.time).dropna('time'))

        iz0 = np.digitize(zpod.values, dataset.depth - 4) - 1
        zbin = xr.DataArray(np.stack([iz0 - 1, iz0, iz0 + 1]),
                            dims=['iz', 'time'],
                            coords={'time': zpod.time, 'iz': [-8, 0, 8]})
        subset = (dataset
                  .sel(depth=dataset.depth[zbin], time=zbin.time))

        if debug:
            if isinstance(dataset, xr.Dataset):
                da = dataset[list(dataset.data_vars)[0]]
            else:
                da = dataset
            plt.figure()
            da.plot(x='time', y='depth', yincrease=False)
            (zpod+zbin.iz).plot.line(x='time', color='k')

        return subset

    def interp_shear(self, kind='ctd', wkb_scale=False):
        '''
        kind: str, optional
            'nearest': interpolate smoothed shear field to χpod depth
            'bins': difference bins about the χpod
            'pod_diff': difference velocity in bins above & below χpod
            'ctd': interpolate u,v to depth of CTDs & difference
            'linear' : fit straight line
        '''

        zpod = (self.zχpod.isel(num=-1, drop=True)
                .interp(time=self.vel.time).dropna('time'))

        if kind == 'nearest':
            uzi = (self.vel[['uz', 'vz', 'u', 'v']]
                   .rolling(depth=4, center=True, min_periods=1).mean()
                   .interp(time=zpod.time, depth=zpod, method='nearest')
                   .interpolate_na('time'))

            uzi.attrs['description'] = ('4 point running mean smoothed '
                                        'central difference shear + '
                                        'nearest neighbour interpolation to '
                                        'χpod depth')
        if kind == 'depth':
            uzi = (self.vel[['u', 'v', 'uz', 'vz']]
                   .sel(depth=136, method='nearest')
                   .interpolate_na('time'))
            uzi.attrs['description'] = 'Shear at 128m depth'

        if kind == 'bins':
            subset = self.sample_along_chipod(self.vel[['u', 'v']])
            # iz0 = np.digitize(zpod.values, self.vel.depth - 4) - 1
            # zbin = xr.DataArray(np.stack([iz0 - 1, iz0, iz0 + 1]),
            #                     dims=['iz', 'time'],
            #                     coords={'time': zpod.time, 'iz': [-8, 0, 8]})
            # subset = (self.vel[['u', 'v']]
            #           .sel(depth=self.vel.depth[zbin], time=zbin.time)
            #           .interpolate_na('time'))
            uzi = (subset
                   .interpolate_na('time')
                   .differentiate('iz')
                   .isel(iz=1, drop=True)
                   .rename({'u': 'uz', 'v': 'vz'}))
            uzi['u'] = subset.u.isel(iz=1)
            uzi['v'] = subset.v.isel(iz=1)

            uzi.attrs['description'] = ('difference bins above, below'
                                        'χpod depth')

        if wkb_scale:
            N = xfilter.lowpass(np.sqrt(self.turb.N2).isel(depth=-1)
                                .interpolate_na('time'),
                                'time', freq=1/30, cycles_per='D')
            wkb_factor = (N/N.mean('time')).interp(time=uzi.time)
            wkb_factor = wkb_factor.ffill('time').bfill('time')

            uzi['u'] = uzi['u'] / np.sqrt(wkb_factor)
            uzi['v'] = uzi['v'] / np.sqrt(wkb_factor)

            uzi['uz'] = uzi['uz'] / (wkb_factor**1.5)
            uzi['vz'] = uzi['vz'] / (wkb_factor**1.5)

            uzi['wkb_factor'] = wkb_factor

        uzi['shear'] = uzi.uz + 1j * uzi.vz

        return uzi.drop('iz', errors="ignore")

    def plot_turb_spectrogram(self):

        tides = dcpy.ts.TidalAliases(1 / 24)
        nfft = 21 * 24  # in hours
        shift = 4 * 24  # in hours

        # get a range of depths that the χpods cover
        # calcualte mean KE over that depth & then wavelet transform
        # meanz = self.zχpod.mean(dim='time')
        # stdz = self.zχpod.std(dim='time')

        # if self.kind == 'ebob':
        #     depth_range = slice((meanz.min()-2*stdz.max()).values,
        #                         (meanz.max()+2*stdz.max()).values)
        # elif self.kind == 'rama':
        #     depth_range = self.vel.u.depth
        # tau = dcpy.ts.Spectrogram(
        #     self.tropflux.tau.sel(time=slice('2013', '2015')).squeeze(),
        #     dim='time', nfft=30, shift=2, multitaper=True, dt=1)
        # tau.freq.attrs['units'] = 'cpd'

        # tau = dcpy.ts.Spectrogram(
        #     self.(time=slice('2013', '2015')).squeeze(),
        #     dim='time',
        #     nfft=60, shift=2, multitaper=True, dt=1)
        # tau.freq.attrs['units'] = 'cpd'

        # label = (str(np.floor(depth_range.start)) + '-'
        #          + str(np.floor(depth_range.stop))
        #          + 'm')

        ax = dict()
        ax['z0'] = dict()
        ax['z1'] = dict()

        lineargs = dict(x='time', yscale='log', lw=0.5, ylim=[1e-12, 1e-1])

        f, axes = plt.subplots(3, 3, sharex=True, constrained_layout=True)
        f.set_size_inches((16, 9))

        ax['z1']['cw'] = axes[0, 1]
        ax['z1']['ccw'] = axes[1, 1]
        ax['z1']['Tz'] = axes[0, 2]
        ax['z1']['N2'] = axes[1, 2]
        ax['z1']['turb'] = axes[2, 1]
        ax['ts'] = axes[2, 2]

        ax['z0']['Tz'] = axes[0, 0]
        ax['z0']['N2'] = axes[1, 0]
        ax['z0']['turb'] = axes[2, 0]

        def add_colorbar(f, ax, hdl):
            # if not hasattr(ax, '__iter__'):
            #     ax = [ax]

            hcbar = f.colorbar(hdl, ax=ax)
            hcbar.formatter.set_powerlimits((0, 0))
            hcbar.update_ticks()

        def plot_spec(ax, spec, name, levels=20):
            var = (spec * spec.freq)
            # var = np.log10(spec)
            hdl = var.plot.contourf(levels=levels, yscale='log',
                                    x='time', robust=True, ax=ax,
                                    cmap=mpl.cm.RdPu, add_colorbar=False)

            var.plot.contour(
                levels=np.linspace(hdl.levels[-1], var.max(), 6)[1:],
                yscale='log', x='time', ax=ax, colors='w',
                linewidths=0.5, add_colorbar=False)

            dcpy.plots.liney([tides['M2'], 2 * tides['M2'],
                              3 * tides['M2'], self.inertial, 1],
                             label=['$M_2$', '$2M_2$', '$3M_2$', '$f_0$', 'd'],
                             lw=1, color='w', ax=ax, zorder=10)

            # (np.sqrt(N2)*86400).plot.line(x='time', color='k', ax=ax,
            #                               add_legend=False)

            ax.text(0.05, 0.05, name,
                    color='k',
                    horizontalalignment='left',
                    transform=ax.transAxes)

            self.MarkSeasonsAndEvents(ax=ax, season=False, zorder=3)

            return hdl

        def interpolate_shear(shear, z):
            newz = (z.interp(time=shear.time)
                    .dropna(dim='time'))
            uzi = xr.DataArray(
                np.diag(shear.interp(depth=newz.values)),
                dims=['time'], coords={'time': newz.time})

            # uzi = shear.interp(time=newz.time, depth=newz)
            return uzi

        hdlcw = []
        hdlccw = []
        hdlN = []
        hdlT = []

        if self.kind == 'ebob':
            for zz in np.arange(len(self.N2.depth)):
                z = self.zχpod.copy().isel(num=zz)
                if zz == 0:
                    z -= 30

                uzi = interpolate_shear(self.vel.uz, z)
                vzi = interpolate_shear(self.vel.vz, z)
                shear = dcpy.ts.Spectrogram((uzi + 1j * vzi)
                                            .interpolate_na(dim='time')
                                            .dropna(dim='time'),
                                            dim='time', nfft=nfft, shift=shift,
                                            multitaper=True, dt=1 / 24)
                shear.freq.attrs['units'] = 'cpd'

                N2 = dcpy.ts.Spectrogram(self.N2.isel(depth=zz)
                                         .dropna(dim='time'),
                                         dim='time', multitaper=True,
                                         dt=1 / 144, nfft=nfft * 6,
                                         shift=shift * 6)

                Tz = dcpy.ts.Spectrogram(self.Tz.isel(depth=zz)
                                         .dropna(dim='time'),
                                         dim='time', multitaper=True,
                                         dt=1 / 144, nfft=nfft * 6,
                                         shift=shift * 6)

                zname = 'z' + str(zz)
                depth = str(self.KT.depth.isel(depth=zz).values) + 'm'

                (self.ε.isel(depth=zz)
                 .resample(time='6H').mean(dim='time')
                 .plot.line(ax=ax[zname]['turb'], **lineargs))

                (self.KT.isel(depth=zz)
                 .resample(time='6H').mean(dim='time')
                 .plot.line(ax=ax[zname]['turb'], **lineargs))

                ax[zname]['turb'].legend(['$ε$', '$K_T$'])
                ax[zname]['turb'].set_ylabel('')

                if zz == 1:
                    hdlcw.append(plot_spec(ax[zname]['cw'], shear.cw,
                                           'CW shear, ' + depth))
                    hdlccw.append(plot_spec(ax[zname]['ccw'], shear.ccw,
                                            'CCW shear, ' + depth,
                                            levels=hdlcw[-1].levels))

                hdlT.append(plot_spec(ax[zname]['Tz'], Tz,
                                      '$dT/dz$, ' + depth))
                hdlN.append(plot_spec(ax[zname]['N2'], N2,
                                      '$N^2$, ' + depth))

            self.tropflux.tau.plot(x='time', ax=ax['ts'], lw=0.5,
                                   color='k')
            if self.ssh is not []:
                ((self.ssh.EKE / 2).plot(ax=ax['ts']))
                (self.ssh.sla.plot(ax=ax['ts']))
                ax['ts'].axhline(0, zorder=-10, color='gray', ls='--')
            ax['ts'].legend(['τ', 'EKE/2', 'SSHA'])
            self.MarkSeasonsAndEvents(ax['ts'])

            # add_colorbar(f, ax['T'], hdlT)

            # zpod = self.zχpod.where(self.zχpod > 10)
            # (self.ctd['T']
            #  .sel(depth2=slice(np.nanmin(zpod)-10,
            #                    np.nanmax(zpod)+10))
            #  .plot.contourf(levels=20, x='time', ax=ax['ts1'],
            #                 cmap=mpl.cm.RdYlBu_r, yincrease=False,
            #                 robust=True))

            # dcpy.plots.liney([depth_range.start, depth_range.stop],
            #                  ax=ax['ts1'], zorder=10)

            # self.ctd.depth[-2, :].plot(ax=ax['ts1'], color='k', lw=0.5)
            # self.PlotχpodDepth(ax=ax['ts1'], lw=0.5)
            # nz = len(Tmean.depth2)
            # (Tmean.isel(depth2=(np.linspace(1, nz-1, 3).astype('int32')))
            #  .plot.line(x='time', ax=ax['ts1'], lw=0.5))

        elif self.kind == 'rama':
            pass

        hdlT[1].levels = hdlT[0].levels
        hdlN[1].levels = hdlN[0].levels

        [aa.set_xlabel('') for aa in axes[:-1, :].flat]
        axes.flat[-1].set_xlim([self.KT.time.min().values,
                                self.KT.time.max().values])
        f.suptitle(self.name)

        for aa in axes.flat:
            self.MarkSeasonsAndEvents(aa)

        for aa in axes[0:2, 1:].flat:
            aa.set_ylabel('', visible=False)

        hdls = dict(cw=hdlcw, ccw=hdlccw,
                    Tz=hdlT, N2=hdlN)

        return ax, hdls

    def plot_mixing_seasons(self, z=104):

        def setup_figure():
            f = plt.figure(constrained_layout=True)
            f.set_size_inches((16, 9))

            if self.kind == 'rama':
                gs = mpl.gridspec.GridSpec(3, 3, figure=f,
                                           height_ratios=[1, 2, 1])
            elif self.kind == 'ebob':
                gs = mpl.gridspec.GridSpec(3, 3, figure=f,
                                           height_ratios=[1, 2, 2])

            ax = dict()
            ax['ts'] = f.add_subplot(gs[0, :])
            ax['ts1'] = ax['ts'].twinx()
            ax['ts1'].spines['right'].set_visible(True)
            ax['vcw'] = f.add_subplot(gs[1, 0])
            ax['vccw'] = f.add_subplot(gs[1, 1], sharex=ax['vcw'])
            ax['T'] = f.add_subplot(gs[1, 2])
            if self.kind == 'ebob':
                ax['scw'] = f.add_subplot(gs[2, 0], sharex=ax['vcw'])
                ax['sccw'] = f.add_subplot(gs[2, 1], sharex=ax['vcw'])
                ax['shear'] = f.add_subplot(gs[2, 2])
            else:
                ax['ts2'] = f.add_subplot(gs[2, :], sharex=ax['ts'])

            return f, ax

        if self.kind == 'ebob':
            uz = (self.vel.uz
                  .sel(depth=z, method='nearest')
                  .interpolate_na(dim='time')
                  .dropna(dim='time'))

            vz = (self.vel.vz
                  .sel(depth=z, method='nearest')
                  .interpolate_na(dim='time')
                  .dropna(dim='time'))
            wz = (uz + 1j * vz)
            wz.name = 'shear'

            # T = ((self.ctd['T'])
            #      .sel(depth2=z, method='nearest')
            #      .interpolate_na(dim='time')
            #      .dropna(dim='time'))

        # else:
            # T = ((self.ctd['T'])
            #      .sel(depth=z, method='nearest')
            #      .interpolate_na(dim='time')
            #      .dropna(dim='time'))

        u = (self.vel.u
             .sel(depth=z, method='nearest')
             .interpolate_na(dim='time')
             .dropna(dim='time'))
        v = (self.vel.v
             .sel(depth=z, method='nearest')
             .interpolate_na(dim='time')
             .dropna(dim='time'))
        w = (u + 1j * v)
        w.name = 'velocity'

        kwargs = dict(multitaper=True, preserve_area=False,
                      linearx=False, lw=0.75)

        for idx, unit in enumerate(self.χpod):
            pod = self.χpod[unit]

            time_range = slice(mdatenum2dt64(pod.time[0]),
                               mdatenum2dt64(pod.time[-1]))

            if len(pod.mixing_seasons) == 0:
                print('No mixing seasons defined for unit '
                      + str(unit))
                continue

            f, ax = setup_figure()

            KT = (self.KT.isel(depth=idx)
                  .sel(time=time_range)
                  .resample(time='6H').mean(dim='time'))
            KT.attrs = self.KT.attrs

            eps = (self.ε.isel(depth=idx)
                   .sel(time=time_range)
                   .resample(time='6H').mean(dim='time'))
            eps.values[eps.values < 1e-12] = 1e-12
            eps.attrs = self.ε.attrs

            KT.plot.line(x='time', ax=ax['ts'], yscale='log', color='k')
            eps.plot.line(x='time', ax=ax['ts1'], yscale='log', color='teal')
            # ax['ts'].legend(['$K_T$', '$ε$'])
            ax['ts'].set_xlabel('')
            ax['ts'].autoscale(enable=True, tight=True)
            ax['ts1'].autoscale(enable=True, tight=True)

            if 'ts2' in ax:
                (self.met.τ.sel(time=time_range)
                 .resample(time='6H').mean(dim='time')
                 .plot(x='time', ax=ax['ts2'], color='k', ylim=[0, 0.3]))

            # iterate over regions
            for region in pod.mixing_seasons.values():
                try:
                    hdl, _ = dcpy.ts.PlotSpectrum(w.sel(time=region),
                                                  dt=_get_dt_in_days(w.time),
                                                  ax=[ax['vcw'], ax['vccw']],
                                                  **kwargs)
                except ValueError:
                    pass

                ax['ts'].axvspan(region.start, region.stop,
                                 facecolor=hdl.get_color(),
                                 alpha=0.1, zorder=-2)

                if 'ts2' in ax:
                    ax['ts2'].axvspan(region.start, region.stop,
                                      facecolor=hdl.get_color(),
                                      alpha=0.1, zorder=-2)

                if 'wz' in locals():
                    dcpy.ts.PlotSpectrum(wz.sel(time=region),
                                         dt=_get_dt_in_days(wz.time),
                                         ax=[ax['scw'], ax['sccw']], **kwargs)

                # dcpy.ts.PlotSpectrum(T.sel(time=region),
                #                      dt=_get_dt_in_days(T.time),
                #                      ax=ax['T'], **kwargs)

                dcpy.ts.PlotSpectrum((eps.sel(time=region)),
                                     dt=_get_dt_in_days(eps.time),
                                     ax=ax['T'], **kwargs)

                # dcpy.ts.PlotSpectrum(np.log10(KT.sel(time=region)),
                #                      dt=_get_dt_in_days(KT.time),
                #                      multitaper=True,
                #                      ax=ax['shear'])

                if 'shear' in self.vel:
                    shear = self.vel.shear.sel(time=region,
                                               depth=slice(10, None))
                    shear = shear.where(shear < np.nanpercentile(shear, 98))
                    mean = (shear.mean(dim='time'))
                    mean.attrs['long_name'] = 'Time-mean shear'
                    mean.attrs['units'] = '1/s'
                    mean.depth.attrs['long_name'] = 'depth'
                    mean.depth.attrs['units'] = 'm'
                    std = self.vel.shear.sel(time=region).std(dim='time')
                    (mean.plot.line(x='depth', ax=ax['shear']))

            ax['vcw'].invert_xaxis()

            for aa in [ax['vcw'], ax['vccw'], ax['T']]:
                txt = aa.get_title()
                aa.text(0.1, 0.1, txt, transform=aa.transAxes)
                aa.set_title('')

            if self.kind == 'ebob':
                for aa in [ax['scw'], ax['sccw']]:
                    txt = aa.get_title()
                    aa.text(0.1, 0.1, txt, transform=aa.transAxes)
                    aa.set_title('')

            axlist = list(ax.values())[2:-1]
            tides = dcpy.ts.TidalAliases(dt=1 / 24)
            dcpy.plots.linex(np.array([1, 2, 3, 4, 5]) * tides['M2'],
                             ax=axlist)
            dcpy.plots.linex(self.inertial, ax=axlist)

            f.suptitle(self.name + ': velocity, T @ ' + str(z) + 'm')

        return ax

    def plot_variances(self, debug=False):
        fM2 = 24 / 12.42

        def interp_var(var):
            z = 110  # self.zχpod.isel(num=1).median().values + 5
            return (var.sel(depth=z, method='nearest')
                    .interpolate_na(dim='time')
                    .dropna('time'))

        def plot_spec(spec, ax=None, levels=20):
            if ax is None:
                _, _ax = plt.subplots()

            var = (spec * spec.freq)
            # var = np.log10(spec)
            hdl = var.plot.contourf(levels=levels, yscale='log',
                                    x='time', robust=True, ax=ax,
                                    cmap=mpl.cm.RdPu, add_colorbar=True)

            var.plot.contour(
                levels=np.linspace(hdl.levels[-1], var.max(), 6)[1:],
                yscale='log', x='time', ax=ax, colors='w',
                linewidths=0.5)

            dcpy.plots.liney([0.8 * fM2, 1.2 * fM2,
                              0.9 * fM2 * 2, 1.1 * fM2 * 2,
                              0.9 * fM2 * 3, 1.1 * fM2 * 3,
                              0.6 * self.inertial, 1.5 * self.inertial],
                             zorder=10, ax=ax)

        kwargs = dict(dim='time', multitaper=True)

        shear_pod = interp_var(self.vel.uz) + 1j * interp_var(self.vel.vz)
        shear_spec = dcpy.ts.Spectrogram(np.abs(shear_pod), **kwargs,
                                         dt=1 / 24, nfft=30 * 24, shift=4 * 24)
        shear_spec.attrs['long_name'] = 'PSD(shear) at 120m'

        # temp_spec = dcpy.ts.Spectrogram(
        #     interp_var(self.ctd['T'].rename({'depth2': 'depth'})),
        #     **kwargs, dt=1/24/6, nfft=30*24*6, shift=4*24*6)
        # temp_spec.attrs['long_name'] = 'PSD(temp)'

        if debug:
            f, ax = plt.subplots(1, 2, sharex=True, sharey=True,
                                 constrained_layout=True)
            plot_spec(shear_spec.sel(time='2014'), ax[0])
            # plot_spec(temp_spec.sel(time='2014'), ax[1])

        dcpy.ts.PlotSpectrum(shear_pod,
                             dt=1 / 24, multitaper=True, decimate=False,
                             twoside=False)
        # dcpy.plots.linex([0.8*fM2, 1.2*fM2,
        #                   0.9*fM2*2, 1.1*fM2*2,
        #                   0.9*fM2*3, 1.1*fM2*3,
        #                   0.6*self.inertial, 1.5*self.inertial],
        #                  zorder=10)

        f0 = self.inertial
        dcpy.plots.linex([f0, 1,
                          f0 + fM2, fM2 - f0,
                          f0 + 2 * fM2, 2 * fM2 - f0,
                          3 * fM2 + f0, 3 * fM2 - f0], zorder=10)

    def plot_turb_fluxes(self, region={}):

        region = self.select_region(region)

        f, ax = plt.subplots(6, 1, sharex=True, constrained_layout=True)

        self.Jq.sel(**region).plot.line(ax=ax[0], hue='depth')
        self.N2.sel(**region).plot.line(ax=ax[1], hue='depth',
                                        add_legend=False)
        self.Js.sel(**region).plot.line(ax=ax[2], hue='depth',
                                        add_legend=False)
        self.ε.sel(**region).plot.line(ax=ax[3], hue='depth', add_legend=False,
                                       yscale='log')
        self.turb.S.sel(**region).plot.line(ax=ax[4], hue='depth',
                                            add_legend=False)

        # shear = (self.calc_shear_bandpass(depth=60)
        #          .resample(time='D').apply(dcpy.util.ms))

        full, low, high, niw, loni = self.filter_interp_shear()

        (low.rolling(time=7*24, center=True).reduce(dcpy.util.ms)
         .sel(**region)
         .plot(ax=ax[-1], label='low', color='k'))
        (niw.rolling(time=7*24, center=True).reduce(dcpy.util.ms)
         .sel(**region).plot(ax=ax[-1], label='$f_0$'))

        # .sel(**region).plot(ax=ax[3], label='$M_2$')
        # shear['M4'].sel(**region).plot(ax=ax[3], label='$M_4$')
        ax[-1].legend()

        ax[0].set_ylim([-150, 10])
        ax[2].set_ylim([0, 5e-2])

        [aa.set_xlabel('') for aa in ax]
        [aa.set_title('') for aa in ax[1:]]
        [self.MarkSeasonsAndEvents(ax=aa) for aa in ax]

    def calc_shear_bandpass(self, depth=120):
        def calc_shear(freqs, depth):
            return (dcpy.ts.BandPassButter(
                self.vel.shear.sel(depth=depth, method='nearest')
                .interpolate_na('time'),
                freqs, dim='time', dt=1 / 24, debug=False))

        shear = xr.Dataset()

        shear['f0'] = calc_shear(np.array([1 / 1.5, 1.5])
                                 * self.inertial.values, depth)
        shear['M2'] = calc_shear(np.array([1 / 1.05, 1.05])
                                 * 24 / 12.42, depth)
        shear['M4'] = calc_shear(np.array([1 / 1.05, 1.05])
                                 * 24 / 12.42 * 2, depth)
        shear['total'] = self.vel.shear.sel(depth=depth, method='nearest')

        return shear

    def isopycnal_shear(self, trange=slice('2014-06-01', '2014-09-01'),
                        zrange=None):
        ''' Maps shear on to isopycnals. Returns Dataset. '''

        if zrange is None:
            zrange = slice(self.zχpod.isel(num=1).median()-10, None)

        T = (self.ctd.ρ_T.sel(time=trange, depth2=zrange)
             .resample(time='H').mean('time')
             .rename({'depth2': 'depth'})
             .dropna('depth', how='all'))
        T['depth'] = T.depth.astype('float32')
        # choose isopycnals
        isos = T.mean('time').isel(depth=slice(0, None, 5))

        isoT = xr.Dataset()
        isoT['depth'] = (xr.DataArray(np.zeros((len(isos), len(T.time))),
                                      dims=['T', 'time'],
                                      coords={'T': isos.values,
                                              'time': T.time})
                         * np.nan)
        isoT['uz'] = xr.zeros_like(isoT.depth) * np.nan
        isoT['vz'] = xr.zeros_like(isoT.depth) * np.nan

        vel = self.vel.sel(time=trange, depth=zrange)

        for tt, t in enumerate(T.time.values):
            subset = T.sel(time=t).dropna('depth', how='all')
            # Si = np.interp(subset.depth.values,
            # self.ctd.depth.sel(time=t, method='nearest').values,
            #                nrl5.ctd.S.sel(time=t, method='nearest').values)
            # rho = sw.pden(Si, subset.T, subset.depth)

            isoT['depth'][:, tt] = np.interp(
                isoT.T.values,
                subset.T.sortby(subset.T).values,
                subset.depth.sortby(subset.T).values,
                left=np.nan, right=np.nan)
            isoT['uz'][:, tt] = np.interp(
                isoT.depth.sel(time=t).values,
                vel.depth.values,
                vel.uz.sel(time=t, method='nearest').values,
                left=np.nan, right=np.nan)
            isoT['vz'][:, tt] = np.interp(
                isoT.depth.sel(time=t).values,
                vel.depth.values,
                vel.vz.sel(time=t, method='nearest').values,
                left=np.nan, right=np.nan)

        return isoT

    def linear_fit_shear(self, region):
        '''
        Estimate shear using linear fits in depth.

        Parameters
        ----------
        region : dict
            dictionary passed to sel.

        Returns
        -------
        shear : xarray.Dataset
            contains uz, vz, shear = np.hypot(uz, vz)
        '''

        u = (self.vel.u.sel(**region)
             .interpolate_na('time')
             .dropna('time', how='any'))
        v = (self.vel.u.sel(**region)
             .interpolate_na('time')
             .dropna('time', how='any'))

        daargs = dict(dims=['time'], coords={'time': u.time})

        uz = xr.DataArray(
            np.apply_along_axis(lambda x: np.polyfit(u.depth, x, 1)[0],
                                u.get_axis_num('depth'), u.values),
            **daargs, name='uz')
        vz = xr.DataArray(
            np.apply_along_axis(lambda x: np.polyfit(u.depth, x, 1)[0],
                                v.get_axis_num('depth'), v.values),
            **daargs, name='vz')
        shear = xr.merge([uz, vz])
        shear['shear'] = np.hypot(uz, vz)

        return shear

    def plot_niw_fraction(self, region=dict(depth=slice(110, 160), time='2014')):

        roll = dict(time=4*24)

        shear = self.linear_fit_shear(region).shear
        # shear = np.sqrt((self.vel.shear.sel(**region)**2).mean('depth'))
        niw_shear = dcpy.ts.BandPassButter(
            shear,
            freqs=np.array([0.7, 1.3]) * self.inertial.values,
            dt=1/24, debug=False)

        ke = (np.hypot(self.vel.u, self.vel.v)**2).sel(**region).mean('depth')
        niw_ke = dcpy.ts.BandPassButter(
            ke,
            freqs=np.array([0.7, 1.3]) * self.inertial.values,
            dt=1/24, debug=False)

        f, ax = plt.subplots(2, 1, constrained_layout=True, sharex=True)

        for full, niw, aa in zip([ke, shear],
                                 [niw_ke, niw_shear],
                                 ax):
            full.rolling(**roll).var().plot(ax=aa)
            niw.rolling(**roll).var().plot(ax=aa)

            a2 = aa.twinx()
            ((niw.rolling(**roll).var() / full.rolling(**roll).var())
             .plot(ax=a2, color='gray'))
            dcpy.plots.set_axes_color(a2, 'gray', spine='right')
            a2.set_ylim([0, 1])
            a2.set_ylabel('Near-inertial fraction')
            aa.set_xlabel('')

            f.suptitle(self.name + ': Rolling 4 day variances')
            ax[0].set_title('KE')
            ax[1].set_title('shear = linear fit to vel in depth = '
                            + str(region['depth']))

        return f, ax

    def debug_salt_flux(self):

        for zz in self.turb.depth:
            f, axx = plt.subplots(5, 1, sharex=True, constrained_layout=True)
            ax = dict(zip(['strat', 'Jq', 'Js', 'eps', 'S'], axx))

            (self.N2*100).sel(depth=zz).plot.line(x='time', ax=ax['strat'])
            self.Tz.sel(depth=zz).plot.line(x='time', ax=ax['strat'])
            # self.Sz.sel(depth=zz).plot.line(x='time', ax=ax['strat'])
            ax['strat'].legend(['$ρ_z$', '$T_z$', '$S_z$'])

            self.Js.sel(depth=zz).plot.line(x='time', ax=ax['Js'])
            self.Jq.sel(depth=zz).plot.line(x='time', ax=ax['Jq'])
            self.KT.sel(depth=zz).plot.line(x='time', ax=ax['eps'], lw=0.5)
            self.turb.S.sel(depth=zz).plot.line(x='time', ax=ax['S'])
            ax['eps'].set_yscale('log')

            [self.MarkSeasonsAndEvents(aa) for aa in axx]
            [aa.set_xlabel('') for aa in axx]

    def filter_interp_shear(self, kind='bins', wkb_scale=False, remove_noise=False):

        filter_kwargs = dict(cycles_per="D", coord="time", order=2)
        lf = self.inertial.values / 2
        hf = self.inertial.values * 2

        if kind == 'filter_then_sample' or kind == "filter_field":
            full = (self.vel[['u', 'v']]
                    .interpolate_na('depth'))
            full = xr.merge([full,
                             full.differentiate('depth')
                             .rename({'u': 'uz', 'v': 'vz'})])
            full['shear'] = full.uz + 1j * full.vz

            # N = xfilter.lowpass(np.sqrt(self.turb.N2).isel(depth=-1)
            #                     .interpolate_na('time'),
            #                     'time', freq=1/30, cycles_per='D')
            # wkb_factor = (N/N.mean('time')).interp(time=uzi.time)
            # wkb_factor = wkb_factor.ffill('time').bfill('time')

            # uzi['u'] = uzi['u'] / np.sqrt(wkb_factor)
            # uzi['v'] = uzi['v'] / np.sqrt(wkb_factor)

            # uzi['uz'] = uzi['uz'] / (wkb_factor**1.5)
            # uzi['vz'] = uzi['vz'] / (wkb_factor**1.5)

            # full['wkb_shear'] = full['shear'] / (wkb_factor ** 1.5)
            # full['wkb_factor'] = wkb_factor

        else:
            uzi = self.interp_shear(kind, wkb_scale=wkb_scale)

            full = xr.Dataset()
            full["shear"] = (uzi.uz.interpolate_na("time")
                             + 1j * uzi.vz.interpolate_na("time"))
            full["u"] = uzi.u
            full["v"] = uzi.v
            full = full.interpolate_na("time")

        full["N2"] = self.turb.N2.isel(depth=1).interp(time=full.time)
        full["Tz"] = self.turb.Tz.isel(depth=1).interp(time=full.time)
        full.attrs['name'] = 'Full'

        low = full.shear.pipe(xfilter.lowpass, freq=1/9, **filter_kwargs)
        low.attrs['name'] = 'LF (< 0.1)'

        res = full.shear - low

        fM2 = res.pipe(xfilter.bandpass,
                       freq=[0.95*(1.93 - self.inertial.values),
                             1.05*(1.93 + self.inertial.values)],
                       **filter_kwargs)

        res -= fM2

        fM4 = res.pipe(xfilter.bandpass,
                       freq=[0.95*(2*1.93 - self.inertial.values),
                             1.05*(2*1.93 + self.inertial.values)],
                       **filter_kwargs)
        res -= fM4
        fM24 = fM2 + fM4

        niw = (res.pipe(xfilter.bandpass,
                        freq=[lf, hf],
                        **filter_kwargs))
        niw.attrs['name'] = 'NIW (6.6d - 2d)'

        res -= niw

        loni = (full.shear.pipe(xfilter.bandpass,
                                freq=[hf, 4],
                                **filter_kwargs)
                - fM2)
        loni.attrs['name'] = 'HF (> 2d)'

        high = res.pipe(xfilter.bandpass, freq=[lf, 4], **filter_kwargs)
        high.attrs['name'] = 'HF (> 6.6d)'

        if remove_noise:
            full['shear'] = full.shear.pipe(xfilter.lowpass, freq=6,
                                            **filter_kwargs)
        if kind == 'filter_then_sample':
            full = self.sample_along_chipod(full).sel(iz=0, drop=True)
            low = self.sample_along_chipod(low).sel(iz=0, drop=True)
            loni = self.sample_along_chipod(loni).sel(iz=0, drop=True)
            niw = self.sample_along_chipod(niw).sel(iz=0, drop=True)
            high = self.sample_along_chipod(high).sel(iz=0, drop=True)

        return full, low, high, niw, loni, fM24

    def compare_shear_spectrum(self):

        import garrettmunk.garrettmunk as gm

        v = ((self.vel.u + 1j * self.vel.v)
             .interpolate_na('depth')
             .interpolate_na('time')
             .dropna('depth', how='any'))

        vel_spec = xrft.power_spectrum(v.sel(depth=slice(120, 500)),
                                       dim=['depth'],
                                       detrend='linear', window=True)
        shear_spec = (2*np.pi * vel_spec.freq_depth)**2 * vel_spec

        self.ctd['N2'] = (9.81/1025 *
                          self.ctd.ρ.differentiate('z')
                          / self.ctd.depth.differentiate('z'))

        (shear_spec.groupby('time.month').mean('time')
         .plot(col='month', col_wrap=4, yscale='log'))

        # gmspec = gm().shear()
