import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd
import seawater as sw
from .accessors import MonsoonAccessor


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
            color='white',
            horizontalalignment='right',
            verticalalignment='center',
            transform=ax.transAxes,
            bbox=dict(facecolor='dimgray', edgecolor=None))


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


def _colorbar(hdl, ax=None):

    if ax is None:
        try:
            ax = hdl.axes
        except AttributeError:
            ax = hdl.ax

    if not isinstance(ax, list):
        box = ax.get_position()

    if isinstance(ax, list) and len(ax) > 2:
        raise ValueError('_colorbar only supports passing 2 axes')

    if isinstance(ax, list) and len(ax) == 2:
        box = ax[0].get_position()
        box2 = ax[1].get_position()

        box.y0 = np.min([box.y0, box2.y0])
        box.y1 = np.max([box.y1, box2.y1])

        box.y0 += box.height * 0.05
        box.y1 -= box.height * 0.05

    axcbar = plt.axes([(box.x0 + box.width)*1.02,
                       box.y0, 0.01, box.height])
    hcbar = plt.colorbar(hdl, axcbar)

    return hcbar


class moor:
    ''' Class for a *single* mooring that has χpods '''

    def __init__(self, lon, lat, name, kind, datadir):

        import collections

        self.name = name
        self.kind = kind
        self.datadir = datadir

        # location
        self.lon = lon
        self.lat = lat

        self.AddSeason()
        self.events = dict()
        self.deployments = []
        self.deploy = dict()

        self.ctd = xr.Dataset()  # TAO CTD
        self.met = xr.Dataset()  # TAO met
        self.tropflux = xr.Dataset()  # tropflux
        self.vel = xr.Dataset()

        # chipods
        self.χpod = collections.OrderedDict()
        self.zχpod = dict([])

        # combined turb data
        self.χ = xr.Dataset()
        self.KT = xr.Dataset()
        self.Jq = xr.Dataset()

        # combine Tz, N² for χpod depths
        self.Tz = xr.Dataset()
        self.N2 = xr.Dataset()

    def __repr__(self):
        import matplotlib.dates as dt

        podstr = ''
        for unit in self.χpod:
            pod = self.χpod[unit]
            podstr += '\t'+pod.name[2:]
            times = (dt.num2date(pod.time[0]).strftime('%Y-%b-%d')
                     + ' → '
                     + dt.num2date(pod.time[-2]).strftime('%Y-%b-%d'))
            podstr += ' | ' + times + '\n'

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

    def CombineTurb(self):
        ''' Combines all χpod χ, KT, Jq into a single DataArray each '''

        χ = []
        KT = []
        Jq = []
        Tz = []
        N2 = []
        z = []

        t = []
        for idx, unit in enumerate(self.χpod):
            pod = self.χpod[unit]
            t.append(pod.time)

        tall = (np.array([[np.nanmin(tt), np.nanmax(tt)] for tt in t]))
        tmatlab = np.arange(np.floor(np.nanmin(tall)),
                            np.ceil(np.nanmax(tall)), 10*60/86400)
        tcommon = ((-86400 + tmatlab * 86400).astype('timedelta64[s]')
                   + np.datetime64('0001-01-01')).astype('datetime64[ns]')

        interpargs = {'left': np.nan, 'right': np.nan}

        for idx, unit in enumerate(self.χpod):
            pod = self.χpod[unit]
            mask = np.logical_not(np.isnan(pod.time))

            z.append(xr.DataArray(
                np.interp(tmatlab, pod.time[mask],
                          pod.depth * np.ones_like(pod.time[mask]),
                          **interpargs)[np.newaxis, :],
                dims=['depth', 'time'],
                coords=[[pod.depth], tcommon], name='zχpod'))

            ρ1 = sw.pden(pod.ctd1.S, pod.ctd1.T, pod.ctd1.z)
            ρ2 = sw.pden(pod.ctd2.S, pod.ctd2.T, pod.ctd2.z)
            ρ = np.interp(tmatlab, pod.ctd1.time, (ρ1+ρ2)/2,
                          **interpargs)[np.newaxis, :]
            T = np.interp(tmatlab, pod.ctd1.time, (pod.ctd1.T+pod.ctd2.T)/2,
                          **interpargs)[np.newaxis, :]
            S = np.interp(tmatlab, pod.ctd1.time, (pod.ctd1.S+pod.ctd2.S)/2,
                          **interpargs)[np.newaxis, :]

            coords = {'z': (['depth', 'time'], z[-1].values),
                      'time': tcommon,
                      'depth': [pod.depth],
                      'lat': self.lat,
                      'lon': self.lon,
                      'ρ': (['depth', 'time'], ρ),
                      'S': (['depth', 'time'], S),
                      'T': (['depth', 'time'], T)}

            # if self.kind == 'ebob':
            #     coords['unit'] = (['depth'], [pod.name[2:5]])

            dims = ['depth', 'time']

            χ.append(xr.DataArray(
                np.interp(tmatlab, pod.time[mask],
                          pod.chi[pod.best]['chi'][mask],
                          **interpargs)[np.newaxis, :],
                dims=dims, coords=coords, name='χ'))

            KT.append(xr.DataArray(
                np.interp(tmatlab, pod.time[mask],
                          pod.KT[pod.best][mask],
                          **interpargs)[np.newaxis, :],
                dims=dims, coords=coords, name='KT'))

            Jq.append(xr.DataArray(
                np.interp(tmatlab, pod.time[mask],
                          pod.Jq[pod.best][mask],
                          **interpargs)[np.newaxis, :],
                dims=dims, coords=coords, name='Jq'))

            Tz.append(xr.DataArray(
                np.interp(tmatlab, pod.time[mask],
                          pod.chi[pod.best]['dTdz'][mask],
                          **interpargs)[np.newaxis, :],
                dims=dims, coords=coords, name='Tz'))

            N2.append(xr.DataArray(
                np.interp(tmatlab, pod.time[mask],
                          pod.chi[pod.best]['N2'][mask],
                          **interpargs)[np.newaxis, :],
                dims=dims, coords=coords, name='N2'))

            # check if there are big gaps (> 1 day)
            # must replace these with NaNs
            time = pod.time[mask]
            dtime = np.diff(pod.time[mask])
            inds = np.where(np.round(dtime) > 0)
            if len(inds[0]) > 0:
                import warnings
                warnings.warn('Found large gap. NaNing out...')
                from dcpy.util import find_approx
                i0 = find_approx(tmatlab, time[inds[0][0]])
                if len(inds[0]) > 1:
                    i1 = find_approx(tmatlab, time[inds[0][1]+2])
                else:
                    i1 = find_approx(tmatlab, time[inds[0][0]+2])

                for var in [χ, KT, Jq, Tz, N2]:
                    var[-1].values[:, i0:i1] = np.nan

        def merge(x0):

            x = xr.merge(map(lambda xx: xx.to_dataset(), x0))[x0[0].name]
            x['depth'] = x.z.mean(dim='time')
            # grouped by depth
            a = [x.sel(depth=zz, drop=False) for zz in np.unique(x['depth'])]
            # concat depth groups in time
            # b = [ for xx in a]

            def merge2(aa):
                if aa.ndim > 3:
                    return xr.merge([aa.isel(depth=zz)
                                    for zz in
                                    np.arange(len(np.atleast_1d(aa.depth)))])
                else:
                    return aa.to_dataset()

            b = [merge2(aa) for aa in a]
            return xr.concat(b, dim='depth')

        self.χ = merge(χ).χ
        self.KT = merge(KT).KT
        self.Jq = merge(Jq).Jq
        self.Tz = merge(Tz).Tz
        self.N2 = merge(N2).N2

        if self.kind == 'ebob':
            z0 = np.interp(tcommon.astype('float32'),
                           self.ctd.time.astype('float32'),
                           self.ctd.depth.isel(z=0))
            z1 = np.interp(tcommon.astype('float32'),
                           self.ctd.time.astype('float32'),
                           self.ctd.depth.isel(z=1))
            self.zχpod = (np.array([[5, 10]]).T
                          + np.array([z0, z1]))

            for a in [self.χ, self.KT, self.Jq, self.Tz, self.N2]:
                a['z'].values = self.zχpod

        else:
            self.zχpod = xr.merge(z).zχpod

    def ReadCTD(self, fname: str, FileType: str='ramaprelim'):

        from scipy.io import loadmat
        import dcpy.util

        if FileType == 'ramaprelim':
            mat = loadmat(fname, squeeze_me=True, struct_as_record=False)
            try:
                mat = mat['rama']
            except KeyError:
                pass

            time = dcpy.util.mdatenum2dt64(mat.time-366)
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
            import netCDF4 as nc
            import matplotlib.dates as dt

            fname = fname + 't' + str(self.lat) + 'n' \
                + str(self.lon) + 'e' + '_10m.cdf'
            f = nc.Dataset(fname)
            t0 = f['time'].units[11:]
            fmt = dt.strpdate2num('%Y-%m-%d %H:%M:%S')
            t0 = fmt(t0)
            self.ctd.Tlong = f['T_20'][:].squeeze()
            self.ctd.Ttlong = f['time'][:] + t0
            f.close()

            fname = '../data/' + 's' + str(self.lat) + 'n' \
                    + str(self.lon) + 'e' + '_hr.cdf'
            f = nc.Dataset(fname)
            t0 = f['time'].units[11:]
            fmt = dt.strpdate2num('%Y-%m-%d %H:%M:%S')
            t0 = fmt(t0)
            self.ctd.Slong = f['S_41'][:].squeeze()
            self.ctd.Stlong = f['time'][:] + t0
            f.close()

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

            time = dcpy.util.mdatenum2dt64(mat['time']-366)
            pres = np.float16(pres)
            z2 = mat2['dbar_dpth']

            self.ctd = xr.Dataset({'S': (['z', 'time'], salt),
                                   'T_S': (['z', 'time'], temp),
                                   'T': (['z2', 'time'], temp2),
                                   'ρ': (['z', 'time'], ρ)},
                                  coords={'depth': (['z', 'time'], pres),
                                          'depth2': ('z2', z2),
                                          'time': ('time', time[0, :])})
            self.ctd['depth'] = self.ctd.depth.fillna(0)

            if fname == 'NRL3':
                # instrument is bad. All salinities are in 20s.
                # Simple offset correction doesn't help
                self.ctd.S.isel(z=3).values.fill(np.nan)

    def ReadMet(self, fname: str=None, WindType='', FluxType=''):

        import airsea as air
        from dcpy.util import mdatenum2dt64

        if WindType == 'pmel':
            if fname is None:
                raise ValueError('I need a filename for PMEL met data!')

            met = xr.open_dataset(fname, autoclose=True)
            spd = met.WS_401.squeeze()
            z0 = abs(met['depu'][0])
            τ = air.windstress.stress(spd, z0, drag='smith')

            self.met = xr.merge([self.met,
                                 xr.DataArray(τ, coords=[met.time.values],
                                              dims=['time'], name='τ')])

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

        P = (xr.open_mfdataset('../ncep/prate*', autoclose=True)
             .sel(lon=self.lon, lat=self.lat, method='nearest').load())

        P = P.rename(dict(time='Ptime', prate='P'))
        # convert from kg/m^2/s to mm/hr
        P *= 1/1000 * 1000 * 3600.0

        self.met = xr.merge([self.met, P])

    def ReadTropflux(self, loc):
        ''' Read tropflux data. Save in moor.tropflux'''

        swr = (xr.open_mfdataset(loc + '/swr_*.nc', autoclose=True)
               .sel(latitude=self.lat, longitude=self.lon, method='nearest')
               .load())
        lwr = (xr.open_mfdataset(loc + '/lwr_*.nc', autoclose=True)
               .sel(latitude=self.lat, longitude=self.lon, method='nearest')
               .load())
        tau = (xr.open_mfdataset(loc + '/tau_*.nc', autoclose=True)
               .sel(latitude=self.lat, longitude=self.lon, method='nearest')
               .load())
        net = (xr.open_mfdataset(loc + '/netflux_*.nc', autoclose=True)
               .sel(latitude=self.lat, longitude=self.lon, method='nearest')
               .load())

        taux = xr.open_mfdataset(loc + '/taux_*.nc', autoclose=True).taux
        tauy = xr.open_mfdataset(loc + '/tauy_*.nc', autoclose=True).tauy
        tx_y = taux.diff(dim='latitude')
        ty_x = tauy.diff(dim='longitude')

        # tropflux is 1° - lets use that to our advantage
        lat2m, _ = sw.dist([self.lat-0.5, self.lat+0.5], self.lon, units='m')
        lon2m, _ = sw.dist(self.lat, [self.lon-0.5, self.lon+0.5], units='m')

        curl = ((ty_x/lon2m - tx_y/lat2m)
                .sel(latitude=self.lat, longitude=self.lon, method='nearest')
                .to_dataset(name='curl'))

        self.tropflux = xr.merge([self.tropflux, swr, lwr, tau, curl, net])

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
        seasons[2014] = {
            'NE': ('2013-Dec-01', '2014-Feb-14'),
            'NESW':  ('2014-Feb-15', '2014-May-05'),
            'SW': ('2014-May-06', '2014-Sep-24'),
            'SWNE': ('2014-Sep-25', '2014-Dec-12')
        }

        seasons[2015] = {
            'NE': ('2014-Dec-12', '2015-Mar-01'),
            'NESW': ('2015-Mar-01', '2015-May-15'),
            'SW': ('2015-May-16', '2015-Oct-14'),
            'SWNE': ('2015-Oct-15', '2015-Dec-01')
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

    def ReadVel(self, fname, FileType: str='ebob'):
        ''' Read velocity data '''

        if FileType == 'pmel':
            self.vel = xr.open_dataset(fname, autoclose=True)

            self.vel.rename({'U_320': 'u',
                             'V_321': 'v'}, inplace=True)
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
            time = dcpy.util.mdatenum2dt64(adcp['date_time']-366).squeeze()
            self.vel = xr.Dataset({'u': (['depth', 'time'], adcp['uu']/100),
                                   'v': (['depth', 'time'], adcp['vv']/100)},
                                  coords={'depth': z, 'time': time})

            dz = self.vel.u.depth.diff(dim='depth')
            uz = self.vel.u.diff(dim='depth')/dz
            vz = self.vel.v.diff(dim='depth')/dz
            shear = np.hypot(uz, vz)
            shear['depth'] = (shear.depth/1.0) - dz/2  # relocate to bin edges
            self.vel['shear'] = shear.rename({'depth': 'depth_shear'})

    def AddChipod(self, name, depth: int,
                  best: str, fname: str='Turb.mat', dir=None):

        import chipy

        if dir is None:
            dir = self.datadir

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
            hdl, lbl, p = \
                    self.χpod[name].SeasonalSummary(ax=ax,
                                                    idx=idx,
                                                    filter_len=filter_len)
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
                size = s0 \
                    + ns * s0 * dn/np.nanstd(dn)

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

        for tt in ax.get_yaxis().get_ticklabels():
            if tt.get_position()[1] < 0:
                tt.set_color(negcolor)

            if tt.get_position()[1] > 0:
                tt.set_color(poscolor)

    def avgplt(self, ax, t, x, flen, filt, axis=-1, decimate=True, **kwargs):
        from dcpy.util import MovingAverage
        from dcpy.ts import BandPassButter

        x = x.copy()

        x_is_xarray = type(x) is xr.core.dataarray.DataArray
        if type(t) is xr.core.dataarray.DataArray:
            dt = (t[3]-t[2]).values
        else:
            dt = (t[3]-t[2]) * 86400

        if np.issubdtype(dt.dtype, np.timedelta64):
            dt = dt.astype('timedelta64[s]').astype('float32')

        if flen is not None:
            if filt == 'mean':
                if x_is_xarray:
                    N = np.int(np.floor(flen/dt))
                    a = x.rolling(time=N, center=True, min_periods=1).mean()
                    a = a.isel(time=slice(N-1, len(a['time'])-N+1, N))
                    t = a['time']
                    return t, a
                else:
                    t = MovingAverage(t.copy(), flen/dt,
                                      axis=axis, decimate=decimate)
                    x = MovingAverage(x.copy(), flen/dt,
                                      axis=axis, decimate=decimate)

            elif filt == 'bandpass':
                flen = np.array(flen.copy())
                assert(len(flen) > 1)
                x = BandPassButter(x.copy(), 1/flen,
                                   dt, axis=axis, dim='time')

            else:
                from dcpy.util import smooth
                if x_is_xarray:
                    xnew = smooth(x.values.squeeze(), flen/dt)
                    x.values = np.reshape(xnew, x.shape)
                else:
                    x = smooth(x, flen/dt, axis=axis)

        if ax is None:
            return t, x
        else:
            hdl = ax.plot(t, x, **kwargs)
            ax.xaxis_date()
            return hdl[0]

    def Quiver(self, t=None, u=None, v=None, ax=None, flen=None,
               filt=None, color='k'):
        import matplotlib.pyplot as plt
        import numpy as np

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

    def MarkSeasonsAndEvents(self, ax=None, season=True, events=True):
        import matplotlib.dates as dt

        if ax is None:
            ax = plt.gca()
        if season:
            seasonColor = {
                'NE': 'beige',
                'NESW': 'lemonchiffon',
                'SW': 'wheat',
                'SWNE': 'honeydew'
            }

            for pp in self.season:
                for ss in self.season[pp]:
                    clr = seasonColor[ss]
                    ylim = ax.get_ylim()
                    xx = dt.date2num(self.season[pp][ss])
                    ax.fill_between(xx, ylim[1], ylim[0],
                                    facecolor=clr, alpha=0.35,
                                    zorder=-10, edgecolor=None)
                    ax.set_ylim(ylim)

        if events:
            for ss in self.events:
                ylim = ax.get_ylim()
                xx = dt.date2num(self.events[ss])
                ax.fill_between(xx, ylim[1], ylim[0],
                                facecolor='palevioletred', alpha=0.35,
                                zorder=-5)
                ax.set_ylim(ylim)

    def PlotCTD(self, name, ax=None, filt=None, filter_len=None,
                add_mld=True, kind='timeseries', lw=1, region={}, **kwargs):

        import dcpy.ts
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
        if self.ctd.depth.ndim > 1 and name is 'S':
            if kind is 'timeseries':
                # compress depth dimension
                var['depth'] = self.ctd.S.depth.median(dim='time')
            else:
                kwargs['x'] = 'time'
                kwargs['y'] = 'depth'

        if self.kind == 'ebob' and name is 'T' and kind is 'timeseries':
            # too many depths to timeseries!
            kind = 'pcolor'

        if kind is 'timeseries':
            from cycler import cycler
            N = len(var.depth)
            colors = mpl.cm.Greys_r(np.arange(N+1)/(N+1))
            ax.set_prop_cycle(cycler('color', colors))

            # more ebob hackery
            if 'z2' in var.dims:
                ydim = 'z2'
            else:
                if 'z' in var.dims:
                    ydim = 'z'
                else:
                    ydim = 'depth'

            hdl = var.plot.line(x='time', hue=ydim, ax=ax, add_legend=False, lw=0.5)

            ncol = N if N < 5 else 5
            ax.legend([str(aa) + 'm'
                       for aa in np.int32(np.round(var.depth))],
                      ncol=ncol)
            ax.set_ylabel(label)

        # if kind is 'profiles':
        #     var -= np.nanmean(var, axis=0)
        #     var += tV
        #     dt = (tV[1, 0] - tV[0, 0]) * 86400.0
        #     if filter_len is not None:
        #         N = np.int(np.ceil(filter_len / dt)) * 2
        #     else:
        #         N = 500

        #     # doesn't work yet
        #     hdl = ax.plot(var.T[::N, :], zV.T[::N, :])

        if kind is 'pcolor' or kind is 'contourf':
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
                                             **kwargs))
                hdl.append(var.plot.contour(ax=ax, levels=20,
                                            colors='k',
                                            linewidths=0.25,
                                            zorder=-1, **kwargs))
                ax.set_ylabel('depth')

        if kind is 'contour':
            hdl = var.plot.contour(ax=ax, levels=20, colors='gray',
                                   linewidths=0.25, zorder=-1,
                                   **kwargs)
            ax.set_ylabel('depth')

        if kind is 'pcolor' or kind is 'contour' or kind is 'contourf':
            if add_mld:
                rho = self.ctd.ρ.resample(time='H').mean(dim='time')
                mld = rho.depth[(np.abs(rho - rho.isel(depth=1)) >
                                 0.005).argmax(axis=0)].drop('depth')
                mld.plot(ax=ax, lw=1, color='darkgray')

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
                self.χ.z.pipe(xfilter, kind='mean', flen=86400)
                .transpose(), color=color, **kwargs)

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
                    raise ValueError(str(name[0])+' not in '+self.name+'.season')

            region = {'time': slice(t0, t1)}

        return region

    def Plotχpods(self, est: str='best', filt='mean', filter_len=86400,
                  quiv=False, TSkind='pcolor', region={},
                  Tlim=[None,None], Slim=[None, None],
                  met='local', fluxvar='netflux', tau='local', event=None):
        ''' Summary plot for all χpods '''

        from dcpy.util import dt64_to_datenum
        import dcpy.plots
        from dcpy.ts import xfilter

        plt.figure(figsize=[11.0, 7.5])
        lw = 0.5

        region = self.select_region(region)

        # initialize axes
        ax = dict()
        ax['met'] = plt.subplot(5, 2, 1)
        ax['N2'] = plt.subplot(5, 2, 3, sharex=ax['met'])

        ax['T'] = plt.subplot(5, 2, 2, sharex=ax['met'])
        if TSkind is not 'timeseries':
            ax['S'] = plt.subplot(5, 2, 4, sharex=ax['met'], sharey=ax['T'])
        else:
            ax['S'] = plt.subplot(5, 2, 4, sharex=ax['met'])

        if self.vel:
            if TSkind is not 'timeseries' and self.kind == 'ebob':
                ax['u'] = plt.subplot(5, 2, 7, sharex=ax['met'], sharey=ax['T'])

            if TSkind is 'timeseries' or self.kind != 'ebob':
                ax['u'] = plt.subplot(5, 2, 7, sharex=ax['met'])

            if self.kind == 'ebob':
                ax['v'] = plt.subplot(5, 2, 9, sharex=ax['met'], sharey=ax['u'])
                ax['shear'] = plt.subplot(5, 2, 5, sharex=ax['met'], sharey=ax['u'])

            else:
                ax['v'] = ax['u']
                ax['χ'] = plt.subplot(5, 2, 9, sharex=ax['met'])
        else:
            ax['χ'] = plt.subplot(5, 2, 7, sharex=ax['met'])

        ax['Tz'] = plt.subplot(5, 2, 6, sharex=ax['met'])
        ax['Kt'] = plt.subplot(5, 2, 8, sharex=ax['met'])
        ax['Jq'] = plt.subplot(5, 2, 10, sharex=ax['met'])

        filtargs = {'kind': filt, 'decimate': True,
                    'flen': filter_len, 'dim': 'time'}
        lineargs = {'x': 'time', 'hue': 'depth', 'linewidth': lw, 'add_legend': False}

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
            region['time'] = slice(t0-dt, t1+dt)
            titlestr += '| Event = ' + event

        plt.suptitle(titlestr, y=1.03)

        # ------------ τ
        if 'τ' not in self.met:
            tau = 'tropflux'

        tauargs = {'color': 'k', 'linewidth': lw, 'zorder': 1,
                   'ax': ax['met']}
        if tau == 'tropflux':
            (self.tropflux.tau
             .pipe(xfilter, **filtargs)
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
                        self.met.P.values/10,
                        flen=filter_len, filt=filt, color='slateblue',
                        linewidth=lw, zorder=-1)

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

        self.PlotFlux(ax['Jq0'], Jq0.sel(**region).time.values, Jq0.sel(**region))
        ax['Jq0'].set_ylabel(fluxvar+' (W/m²)', labelpad=0)
        ax['Jq0'].spines['right'].set_visible(True)
        ax['Jq0'].spines['left'].set_visible(False)
        ax['Jq0'].xaxis_date()

        if filt == 'bandpass':
            ax['Jq0'].set_ylim(
                np.array([-1, 1]) * np.max(np.abs(ax['Jq0'].get_ylim())))

        ((self.N2.copy()
          .pipe(xfilter, **filtargs)
          .sel(**region)/1e-4)
         .plot.line(ax=ax['N2'], **lineargs))

        (self.Tz.copy()
         .pipe(xfilter, **filtargs)
         .sel(**region)
         .plot.line(ax=ax['Tz'], **lineargs))

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
                    lw=0.5, region=region, add_colorbar=False)
        ax['Splot'] = self.PlotCTD('S', ax['S'], vmin=Slim[0], vmax=Slim[1],
                                   **ctdargs)
        ax['Tplot'] = self.PlotCTD('T', ax['T'], vmin=Tlim[0], vmax=Tlim[1],
                                   **ctdargs)

        ax['met'].set_ylabel('$τ$ (N/m²)')

        ax['N2'].set_title('')
        ax['N2'].legend([str(zz)+' m' for zz in self.N2.depth.values])
        ax['N2'].set_ylabel('$N²$ ($10^{-4}$)')
        limy = ax['N2'].get_ylim()
        if filt != 'bandpass':
            ax['N2'].set_ylim([0, limy[1]])

        ax['Tz'].set_title('')
        ax['Tz'].set_ylabel('$\partial T/ \partial z$')
        ax['Tz'].axhline(0, color='gray', zorder=-1, linewidth=0.5)
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
                    ax['u'].legend(('$u_{'+str(zint)+'}$',
                                    '$v_{'+str(zint)+'}$'), ncol=2)
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

                    vargs['vmin'] = -1*np.max(np.abs([mn, mx]))
                    vargs['vmax'] = np.max(np.abs([mn, mx]))

                    ax['Uplot'] = uplt.plot.contourf(ax=ax['u'], **vargs)
                    ax['Vplot'] = vplt.plot.contourf(ax=ax['v'], **vargs)

                    labelargs = dict(x=0.05, y=0.15, alpha=0.05)
                    _corner_label('u', **labelargs, ax=ax['u'])
                    _corner_label('v', **labelargs, ax=ax['v'])
                    self.PlotχpodDepth(ax=ax['u'], color='k')
                    self.PlotχpodDepth(ax=ax['v'], color='k')

                    ax['u'].set_ylim(ax['T'].get_ylim())
                    ax['v'].set_ylim(ax['T'].get_ylim())

        if 'shear' in ax:
            shhdl = (self.vel.shear.sel(**region)
                     .plot.contourf(x='time', ax=ax['shear'],
                                    add_colorbar=False, yincrease=False,
                                    robust=True))
            ax['shear'].set_ylabel('depth')
            ax['shear'].set_ylim(ax['T'].get_ylim())
            self.PlotχpodDepth(ax=ax['shear'], color='k')
            _corner_label('|$(u_z, v_z)$|', x=0.15, y=0.15, ax=ax['shear'])

        ax['met'].set_xlim([self.χ.sel(**region).time.min().values,
                            self.χ.sel(**region).time.max().values])
        plt.gcf().autofmt_xdate()

        for name in ['N2', 'T', 'S', 'v', 'χ', 'Kt', 'Jq', 'Tz']:
            if name in ax:
                self.MarkSeasonsAndEvents(ax[name])
                if filt == 'bandpass':
                    if (name not in ['T', 'S'] or
                            (name in ['T', 'S'] and TSkind == 'timeseries')):
                        dcpy.plots.liney(0, ax=ax[name])

        self.MarkSeasonsAndEvents(ax['met'], season=False)

        plt.tight_layout(w_pad=5, h_pad=-0.5)

        hcbar = dict()
        if isinstance(ax['Tplot'][0], mpl.contour.QuadContourSet):
            hcbar['T'] = _colorbar(ax['Tplot'][0])

        if (isinstance(ax['Splot'][0], mpl.contour.QuadContourSet)
                or isinstance(ax['Splot'][0], mpl.collections.PathCollection)):
            hcbar['S'] = _colorbar(ax['Splot'][0])
            if ax['S'].get_ylim()[0] > 300:
                ax['S'].set_ylim([150, 0])

        if 'Uplot' in ax and 'Vplot' in ax and self.kind == 'ebob':
            hcbar['uv'] = _colorbar(ax['Uplot'], [ax['u'], ax['v']])

        if 'shear' in ax:
            hcbar['shear'] = _colorbar(shhdl, ax=ax['shear'])

        # ax['cbar'].set_ylabel(colorlabel)

        return ax

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

        import matplotlib.pyplot as plt
        import numpy as np
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
            τ = BandPassButter(τ, freqs=freqs, dt=(tτ[3]-tτ[2])*86400.0)
            titlestr += str(1/freqs/86400.0) + ' day bandpassed '

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
                                     dt=(tJ[3]-tJ[2])*86400.0)

            τi = np.interp(tJ, tτ, τ)

            if np.all(np.isnan(Jqt)):
                raise ValueError('Jqt is all NaN. Did you filter too much?')

            # calculate and plot cross-correlations
            plt.axes(ax.ravel()[idx])
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
        plt.xlabel(metvar+" lag (days)")

        plt.tight_layout()

    def PlotAllSpectra(self, filter_len=None, nsmooth=5,
                       SubsetLength=None, ticks=None, **kwargs):

        import matplotlib.pyplot as plt

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

    def TSPlot(self, pref=0, ax=None,
               varname='KT', varmin=1e-3, filter_len=86400):
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

        import seawater as sw
        import matplotlib.pyplot as plt
        import numpy as np

        size = 5

        if ax is None:
            ax = plt.gca()

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        markers = '+^s'

        for idx, unit in enumerate(self.χpod):
            pod = self.χpod[unit]

            var, name, _, _ = pod.ChooseVariable(varname)
            # daily average quantities
            t, var = pod.FilterEstimate(
                'mean', pod.time, var, filter_len=filter_len, decimate=True)

            t1, T1 = pod.FilterEstimate(
                'mean',
                pod.ctd1.time,
                pod.ctd1.T,
                filter_len=filter_len,
                decimate=True)
            _, S1 = pod.FilterEstimate(
                'mean',
                pod.ctd1.time,
                pod.ctd1.S,
                filter_len=filter_len,
                decimate=True)

            t2, T2 = pod.FilterEstimate(
                'mean',
                pod.ctd2.time,
                pod.ctd2.T,
                filter_len=filter_len,
                decimate=True)
            _, S2 = pod.FilterEstimate(
                'mean',
                pod.ctd2.time,
                pod.ctd2.S,
                filter_len=filter_len,
                decimate=True)

            # interpolate onto averaged χpod time grid
            T1 = np.interp(t, t1, T1)
            S1 = np.interp(t, t1, S1)
            T2 = np.interp(t, t2, T2)
            S2 = np.interp(t, t2, S2)

            mask = var > varmin
            frac = sum(mask) / len(mask)
            if frac < 0.1:
                alpha = 0.6
            else:
                alpha = 0.3

            ax.plot(
                np.concatenate([S1[mask], S2[mask]]),
                np.concatenate([T1[mask], T2[mask]]),
                color='k',
                linestyle='None',
                label=pod.name,
                marker=markers[idx],
                alpha=alpha,
                zorder=2)

        for index, z in enumerate(self.ctd.depth):
            S = self.ctd.sal[:, index]
            T = self.ctd.temp[:, index]
            ax.scatter(
                S[::10],
                T[::10],
                s=size,
                facecolor=colors[index],
                alpha=0.1,
                label='CTD ' + str(np.int(z)) + ' m',
                zorder=-1)

        # make sure legend has visible entries
        hleg = plt.legend()
        for hh in hleg.legendHandles:
            hh.set_alpha(1)

        Slim = ax.get_xlim()
        Tlim = ax.get_ylim()
        Tvec = np.arange(Tlim[0], Tlim[1], 0.1)
        Svec = np.arange(Slim[0], Slim[1], 0.1)
        [Smat, Tmat] = np.meshgrid(Svec, Tvec)

        # density contours
        ρ = sw.pden(Smat, Tmat, pref) - 1000
        cs = ax.contour(
            Smat, Tmat, ρ, colors='gray', linestyles='dashed', zorder=-1)
        ax.clabel(cs, fmt='%.1f')

        # labels
        ax.set_xlabel('S')
        ax.set_ylabel('T')
        ax.set_title(self.name)

        ax.annotate(
            name + '$^{' + str(np.round(filter_len / 86400, decimals=1)) +
            ' d}$ > ' + str(varmin),
            xy=[0.75, 0.9],
            xycoords='axes fraction',
            size='large',
            bbox=dict(facecolor='gray', alpha=0.15, edgecolor='none'))

    def PlotCoherence(self, ax, v1, v2, nsmooth=5, multitaper=True):
        import dcpy.ts
        import dcpy.plots

        if multitaper:
            f, Cxy, phase, siglevel = \
                          dcpy.ts.MultiTaperCoherence(v1, v2,
                                                      dt=1, tbp=nsmooth)
            siglevel = siglevel[0]
        else:
            f, Cxy, phase, siglevel = \
                      dcpy.ts.Coherence(v1, v2, dt=1, nsmooth=nsmooth)

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

        import numpy as np
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
        import numpy as np

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

        import matplotlib.pyplot as plt
        import dcpy.ts

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
            if metvar is '':
                continue

            if metvar is 'Jq':
                met = self.met.Jq0
                tmet = self.met.Jtime
                axes = [ax2, ax3]
                label = '$J_q^0$'
                t, m = self.avgplt(None, tmet, met, filter_len, filt)
                ax0.plot(t, m, 'k', label=label)
                self.PlotFlux(ax0, t, m, alpha=0.1)
                dcpy.ts.PlotSpectrum(met, ax=ax1, color='k')

            if metvar is 'wind':
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
                                label='$J_q^t$'+pod.name[5:])
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

    def Budget(self, z=15):

        import dcpy.ts
        import dcpy.plots

        # from statsmodels.nonparametric.smoothers_lowess import lowess
        # import dcpy.plots

        ρ = 1025
        cp = 4000

        resample_args = {'time': '1H'}

        # T = dcpy.ts.xfilter(self.ctd['T'], kind='hann', flen=3*3600.0)
        T = self.ctd['T'].resample(**resample_args).mean(dim='time')
        rho = self.ctd.ρ.resample(**resample_args).mean(dim='time')

        # mixed layer depth
        mld = rho.depth[(np.abs(rho - rho.isel(depth=1)) >
                         0.01).argmax(axis=0)].drop('depth')

        # interpolate temp to 1m grid, then calculate heat content
        f = sp.interpolate.RectBivariateSpline(T.time.astype('float32'),
                                               T.depth, T.values.T,
                                               kx=1, ky=1)
        idepths = np.arange(1, 100, 1)
        Ti = xr.DataArray(f(T.time.astype('float32'), idepths),
                          dims=['time', 'depth'],
                          coords=[T.time, idepths])
        Q = ρ * cp * xr.apply_ufunc(sp.integrate.cumtrapz,
                                    Ti, Ti.depth,
                                    kwargs={'axis': -1})
        Q['depth'] = Q.depth[:-1]


        def interp_b_to_a(a, b):
            return np.interp(a.time.astype(np.float32),
                        b.time.astype(np.float32),
                        b.squeeze(),
                        left=np.nan, right=np.nan)

        # budget is dQ/dt = Jq0 - Ih + Jqt
        # where both Jqs are positive when they heat the surface
        # i.e. Jqt > 0 → heat is moving upward
        Jq0 = xr.DataArray(interp_b_to_a(Q, self.flux.Jq0),
                           dims=['time'], coords=[Q.time])

        swr = xr.DataArray(interp_b_to_a(Q, self.flux.swr),
                           dims=['time'], coords=[Q.time])

        # penetrative heating that leaves mixed layer
        Ih = 0.45 * swr * np.exp(-0.04*mld)

        Jqt = self.Jq
        Jqt.values[Jqt.values == 0] = np.nan
        Jqt = Jqt.resample(**resample_args).mean(dim='time')

        sfcflx = (Jq0 - Ih).resample(**resample_args).mean(dim='time')
        dJdz = ((sfcflx + Jqt)/Jqt.depth).where(Jqt.time == Jq0.time)
        dJdz.name = 'dJdz'

        # Qt = (dQdt.resample(**resample_args)
        #       .mean(dim='time')
        #       .sel(depth=Jqt.depth)
        #       .where(dQdt.time == Jq0.time))
        # Qt.name = 'Qt'

        def average(x):
            return x.resample(time='W').mean(dim='time')
        def ddt(x):

            dt = x.time.diff(dim='time')/np.timedelta64(1, 's')
            return x.diff(dim='time')/dt

        dQdt = Q.pipe(average).pipe(ddt)


        f, ax = plt.subplots(2, 1, sharex=True)
        plt.axes(ax[0])
        (Jqt.sel(depth=z).where(mld <= z)
         .pipe(average)
         .plot(x='time', label='Jqt'))
        (sfcflx
         .pipe(average).where(mld <= z)
         .plot(x='time', label='ML heating'))
        (dQdt.sel(depth=z).where(mld <= z)
         .plot.line(x='time', color='k', label='dQ/dt'))
        ((sfcflx.pipe(average)
          + Jqt.sel(depth=z).pipe(average))
         .plot.line('k--', x='time', label='ML heating+Jqt'))

        # ((sfcflx + Jqt.sel(depth=z)).pipe(average)
        #  .plot.line('lightgray', lw=2, x='time', label='ML heating+Jqt'))
        ax[0].legend()
        xlim = ax[0].get_xlim()
        dcpy.plots.liney(0)

        (np.hypot(self.vel.u,
                  self.vel.v)
         .resample(time='W').mean(dim='time')
         .plot(ax=ax[1]))
        ax[0].set_xlim(xlim)
        ax[0].set_ylim([-150, 150])

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
