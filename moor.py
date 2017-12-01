class moor:
    ''' Class for a *single* mooring that has χpods '''

    def __init__(self, lon, lat, name, datadir):

        import collections
        import xarray as xr

        self.name = name
        self.datadir = datadir

        # location
        self.lon = lon
        self.lat = lat

        # "special" events
        self.special = dict()
        self.season = dict()

        self.ctd = xr.Dataset()  # TAO CTD
        self.met = xr.Dataset()  # TAO met
        self.tropflux = xr.Dataset()  # tropflux
        self.vel = xr.Dataset()

        # chipods
        self.χpod = collections.OrderedDict()
        self.zχpod = collections.OrderedDict()

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
        for ss in self.special:
            specstr += ('\t' + ss + ' | '
                        + self.special[ss][0].strftime('%Y-%b-%d')
                        + ' → '
                        + self.special[ss][1].strftime('%Y-%b-%d')
                        + '\n')

        specstr = specstr[1:]  # remove initial tab

        return ('mooring ' + self.name
                + '\nχpods: ' + podstr
                + '\nEvents: ' + specstr)

    def __str__(self):
        return self.name + ' mooring'

    def CombineTurb(self):
        ''' Combines all χpod χ, KT, Jq into a single DataArray each '''
        import xarray as xr
        import numpy as np
        import pandas as pd
        import matplotlib.dates as mdt

        χ = []
        KT = []
        Jq  = []
        Tz = []
        N2 = []

        for idx, unit in enumerate(self.χpod):
            pod = self.χpod[unit]

            mask = np.logical_not(np.isnan(pod.time))
            times = ((-86400 + pod.time[mask]*86400).astype('timedelta64[s]')
                     + np.datetime64('0001-01-01'))

            χ.append(xr.DataArray(pod.chi[pod.best]['chi'][np.newaxis,mask],
                                  coords=[[pod.depth], times],
                                  dims=['depth', 'time'],
                                  name='χ'))

            KT.append(xr.DataArray(pod.KT[pod.best][np.newaxis,mask],
                                   coords=[[pod.depth], times],
                                   dims=['depth', 'time'], name='KT'))

            Jq.append(xr.DataArray(pod.Jq[pod.best][np.newaxis,mask],
                                   coords=[[pod.depth], times],
                                   dims=['depth', 'time'], name='Jq'))

            Tz.append(xr.DataArray(pod.chi[pod.best]['dTdz'][np.newaxis,mask],
                                  coords=[[pod.depth], times],
                                  dims=['depth', 'time'],
                                  name='Tz'))

            N2.append(xr.DataArray(pod.chi[pod.best]['N2'][np.newaxis,mask],
                                  coords=[[pod.depth], times],
                                  dims=['depth', 'time'],
                                  name='N2'))


        ds = xr.merge(χ)
        self.χ = ds.χ.resample('10min', dim='time', how='mean')

        ds = xr.merge(KT)
        self.KT = ds.KT.resample('10min', dim='time', how='mean')

        ds = xr.merge(Jq)
        self.Jq = ds.Jq.resample('10min', dim='time', how='mean')

        ds = xr.merge(Tz)
        self.Tz = ds.Tz.resample('10min', dim='time', how='mean')

        ds = xr.merge(N2)
        self.N2 = ds.N2.resample('10min', dim='time', how='mean')


    def ReadCTD(self, fname: str, FileType: str='ramaprelim'):

        import seawater as sw
        import numpy as np
        import xarray as xr
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
                                          'time': ('time', time[0,:])})
            self.ctd['depth'] = self.ctd.depth.fillna(0)

    def ReadMet(self, fname: str=None, WindType='', FluxType=''):

        import airsea as air
        import matplotlib.dates as dt
        import numpy as np
        import xarray as xr

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
            import matplotlib.dates as dt
            mat = loadmat(fname, squeeze_me=False)
            # self.met.Jq0 = -mat['Jq']['swf'][0][0][0]
            self.met.Jq0 = -mat['Jq']['nhf'][0][0][0]
            self.met.Jtime = mat['Jq']['t'][0][0][0] - 366
            self.met.swr = xr.DataArray(-mat['Jq']['swf'][0][0][0],
                                        coords=[dt.num2date(self.met.Jtime)],
                                        dims=['time'], name='shortwave rad.')

        if FluxType == 'precip':
            met = nc.MFDataset('../ncep/prate*')
            lat = met['lat'][:]
            lon = met['lon'][:]
            time = met['time'][:]
            self.met.P = interpn((time, np.flipud(lat), lon),
                                 np.fliplr(met['prate'][:, :, :]),
                                 (time, self.lat, self.lon))
            self.met.Ptime = time/24.0 \
                + dt.date2num(dt.datetime.date(1800, 1, 1))

            # convert from kg/m^2/s to mm/hr
            self.met.P *= 1/1000 * 1000 * 3600.0
            met.close()

    def ReadTropflux(self, loc):
        ''' Read tropflux data. Save in moor.tropflux'''

        import xarray as xr
        import seawater as sw

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

    def AddSpecialTimes(self, pods, name, t0, t1):
        import datetime as pdt

        for pp in pods:
            unit = self.χpod[pp]

            try:
                unit.special[name] = [
                    pdt.datetime.strptime(t0, '%Y-%b-%d'),
                    pdt.datetime.strptime(t1, '%Y-%b-%d')
                ]
            except:
                unit.special[name] = [
                    pdt.datetime.strptime(t0, '%Y-%m-%d'),
                    pdt.datetime.strptime(t1, '%Y-%m-%d')
                ]

        # append to the mooring list
        self.special[name] = unit.special[name]

    def AddSeason(self, pods, name, t0, t1):
        import datetime as pdt

        for pp in pods:
            unit = self.χpod[pp]
            try:
                unit.season[name] = [
                    pdt.datetime.strptime(t0, '%Y-%b-%d'),
                    pdt.datetime.strptime(t1, '%Y-%b-%d')
                ]
            except:
                unit.season[name] = [
                    pdt.datetime.strptime(t0, '%Y-%m-%d'),
                    pdt.datetime.strptime(t1, '%Y-%m-%d')
                ]

            self.season[pp] = unit.season

    def ReadVel(self, fname, FileType: str='ebob'):
        ''' Read velocity data '''

        import xarray as xr

        if FileType == 'pmel':
            import numpy as np

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

            adcp = loadmat('../ancillary/adcp/' + self.name + '.mat')
            import IPython; IPython.core.debugger.set_trace()

            z = adcp['depth_levels'].squeeze()
            time = dcpy.util.mdatenum2dt64(adcp['date_time']-366).squeeze()
            self.vel = xr.Dataset({'u': (['depth', 'time'], adcp['u']/100),
                                   'v': (['depth', 'time'], adcp['v']/100)},
                                  coords={'depth': z, 'time': time})


    def AddChipod(self, name, depth: int,
                  best: str, fname: str='Turb.mat', dir=None):

        import sys
        if 'home/deepak/python' not in sys.path:
            sys.path.append('/home/deepak/python')

        import chipy

        if dir is None:
            dir = self.datadir

        self.χpod[name] = chipy.chipod(
            dir + '/data/', str(name), fname, best, depth=depth)
        self.zχpod[name] = depth

    def SetColorCycle(self, ax):

        import matplotlib.pyplot as plt
        import numpy as np

        z = np.array(list(self.zχpod.values()))
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

        import numpy as np
        import matplotlib.pyplot as plt
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
        import xarray as xr
        from dcpy.util import MovingAverage
        from dcpy.ts import BandPassButter
        import numpy as np

        x = x.copy()

        x_is_xarray = type(x) is xr.core.dataarray.DataArray
        if type(t) is xr.core.dataarray.DataArray:
            dt = (t[3]-t[2]).values.astype('timedelta64[s]').astype('float32')
        else:
            dt = (t[3]-t[2]) * 86400

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

    def MarkSeasonsAndSpecials(self, ax, season=True, special=True):
        import matplotlib.dates as dt

        if season:
            seasonColor = {
                'NE': 'beige',
                'NE→SW': 'lemonchiffon',
                'SW': 'wheat',
                'SW→NE': 'honeydew'
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

        if special:
            for ss in self.special:
                ylim = ax.get_ylim()
                xx = dt.date2num(self.special[ss])
                ax.fill_between(xx, ylim[1], ylim[0],
                                facecolor='palevioletred', alpha=0.35,
                                zorder=-5)
                ax.set_ylim(ylim)

    def PlotCTD(self, name, ax=None, filt=None, filter_len=None,
                kind='timeseries', lw=1, t0=None, t1=None, **kwargs):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np
        import dcpy.ts

        N = 6
        if ax is None:
            ax = plt.gca()

        cmap = plt.get_cmap('RdBu_r')

        if kind is 'pcolor' or kind is 'contour' or kind is 'contourf':
            # filter before subsetting
            var = dcpy.ts.xfilter(self.ctd[name], dim='time',
                                  kind=filt, flen=filter_len, decimate=False)

        if t0 is not None and t1 is not None:
            from dcpy.util import find_approx
            it0 = find_approx(tV[:, 0], t0)
            it1 = find_approx(tV[:, 0], t1)

            var = var[it0:it1, :]
            zV = zV[it0:it1, :]
            tV = tV[it0:it1, :]

        label = '$' + self.ctd[name].name + '$'

        if self.ctd.depth.ndim > 1 and name is 'S':
            kwargs['x'] = 'time'
            kwargs['y'] = 'depth'

        if kind is 'timeseries':
            from cycler import cycler
            colors = mpl.cm.Greys_r(np.arange(N+1)/(N+1))
            ax.set_prop_cycle(cycler('color', colors))

            hdl = self.avgplt(
                ax, tV[:, 0], var, filter_len, filt, axis=0, linewidth=lw)
            ax.legend([str(aa) + 'm'
                       for aa in np.int32(np.round(self.ctd.depth[:-1]))],
                      ncol=N)
            ax.set_ylabel(label)

        if kind is 'profiles':
            var -= np.nanmean(var, axis=0)
            var += tV
            dt = (tV[1, 0] - tV[0, 0]) * 86400.0
            if filter_len is not None:
                N = np.int(np.ceil(filter_len / dt)) * 2
            else:
                N = 500

            # doesn't work yet
            hdl = ax.plot(var.T[::N, :], zV.T[::N, :])

        if kind is 'pcolor' or kind is 'contourf':
            hdl = []
            hdl.append(var.plot.contourf(ax=ax, levels=25,
                                         cmap=cmap, zorder=-1,
                                         **kwargs))
            hdl.append(var.plot.contour(ax=ax, levels=20,
                                        colors='gray',
                                        linewidths=0.25,
                                        zorder=-1, **kwargs))
            ax.set_ylabel('depth')
            ax.invert_yaxis()

        if kind is 'contour':
            hdl = var.plot.contour(ax=ax, levels=20, colors='gray',
                                   linewidths=0.25, zorder=-1,
                                   **kwargs)
            ax.set_ylabel('depth')

        if kind is 'pcolor' or kind is 'contour' or kind is 'contourf':
            # label in top-right corner
            ax.text(0.95, 0.9, label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    bbox=dict(facecolor='k', alpha=0.05))

            # showing χpod depth
            [ax.axhline(z, color='gray', linewidth=0.5) for z in self.χ.depth]

        return hdl

    def Plotχpods(self, est: str='best', filt='mean', filter_len=86400,
                  pods=[], quiv=True, TSkind='timeseries', t0=None, t1=None,
                  met='local', fluxvar='netflux', tau='local'):
        ''' Summary plot for all χpods '''

        import matplotlib.pyplot as plt
        import numpy as np
        from dcpy.util import dt64_to_datenum
        import dcpy.plots
        from dcpy.plots import offset_line_plot
        from dcpy.ts import xfilter

        plt.figure(figsize=[12.5, 6.5])
        lw = 0.5

        # initialize axes
        ax = dict()
        ax['met'] = plt.subplot(4, 2, 1)
        ax['N2'] = plt.subplot(4, 2, 3, sharex=ax['met'])
        ax['T'] = plt.subplot(4, 2, 5, sharex=ax['met'])
        if self.vel and self.vel.u is not [] and quiv:
            ax['v'] = plt.subplot(4, 2, 7, sharex=ax['met'])
        else:
            ax['χ'] = plt.subplot(4, 2, 7, sharex=ax['met'])
        # ax['T'] = plt.subplot2grid((4, 2), (2, 0),
        #                            rowspan=2, sharex=ax['met'])
        # ax['T'].rowNum = 3

        ax['S'] = plt.subplot(4, 2, 2, sharex=ax['met'])
        ax['Tz'] = plt.subplot(4, 2, 4, sharex=ax['met'])
        ax['Kt'] = plt.subplot(4, 2, 6, sharex=ax['met'])
        ax['Jq'] = plt.subplot(4, 2, 8, sharex=ax['met'])

        filtargs = {'kind': filt, 'decimate': True,
                    'flen': filter_len, 'dim': 'time'}
        plotargs = {'linewidth': lw, 'legend': False}

        if filter_len is None:
            filt = None

        if filter_len is not None:
            ax['met'].set_title(self.name + ' | '
                                + self.GetFilterLenLabel(filt, filter_len))
        else:
            ax['met'].set_title(self.name)

        # ------------ τ
        ax['met'].set_clip_on(False)
        if 'τ' not in self.met:
            tau = 'tropflux'

        if tau == 'tropflux':
            self.avgplt(ax['met'],
                        dt64_to_datenum(self.tropflux.tau.time.values),
                        self.tropflux.tau.values,
                        filter_len, filt, color='k',
                        linewidth=lw, zorder=1)
        else:
            self.avgplt(ax['met'], self.met.τ.time, self.met.τ,
                        filter_len, filt, color='k',
                        linewidth=lw, zorder=1)

        if filt == 'bandpass':
            ax['met'].set_ylim([-0.1, 0.1])
        else:
            ax['met'].set_ylim([0, 0.3])

        # ----------- precip
        if 'P' in self.met:
            self.avgplt(ax['met'], self.met.P.time, self.met.P/10,
                        flen=filter_len, filt=filt, color='slateblue',
                        linewidth=lw, zorder=-1)

        # ------------ flux
        ax['Jq0'] = ax['met'].twinx()
        ax['Jq0'].set_zorder(-1)

        if 'Jq0' not in self.met:
            # no mooring flux
            met = 'tropflux'

        if met == 'tropflux':
            datenum = dt64_to_datenum(self.tropflux.time.values)
            time, Jq = self.avgplt(None, datenum,
                                   self.tropflux[fluxvar].values,
                                   filter_len, filt)
            ax['Jq0'].set_ylabel(fluxvar+' (W/m²)', labelpad=0)
            self.PlotFlux(ax['Jq0'], time, Jq)

        elif 'Jq0' in self.met:
            time, Jq = self.avgplt(None, self.met.Jtime, self.met.Jq0,
                                   filter_len, filt)
            ax['Jq0'].set_ylabel('$J_q^0$ (W/m²)', labelpad=0)
            self.PlotFlux(ax['Jq0'], time, Jq)

        ax['Jq0'].spines['right'].set_visible(True)
        ax['Jq0'].spines['left'].set_visible(False)
        ax['Jq0'].xaxis_date()

        if filt == 'bandpass':
            ax['Jq0'].set_ylim(
                np.array([-1, 1]) * np.max(np.abs(ax['Jq0'].get_ylim())))

        offset_line_plot(self.N2.copy().pipe(xfilter, **filtargs)/1e-4,
                         x='time', y='depth', remove_mean=False,
                         offset=0, ax=ax['N2'], **plotargs)

        offset_line_plot(self.Tz.copy().pipe(xfilter, **filtargs),
                         x='time', y='depth', remove_mean=False,
                         offset=0, ax=ax['Tz'], **plotargs)

        # ---------- χpods
        labels = []
        xlim = [1e6, 0]
        if pods == []:
            pods = list(self.χpod.keys())

        if 'χ' in ax:
            offset_line_plot(self.χ.copy().pipe(xfilter, **filtargs),
                             x='time', y='depth', remove_mean=False,
                             offset=0, ax=ax['χ'], **plotargs)
            ax['χ'].set_yscale('log')

        offset_line_plot(self.KT.copy().pipe(xfilter, **filtargs),
                         x='time', y='depth', remove_mean=False,
                         offset=0, ax=ax['Kt'], **plotargs)
        ax['Kt'].set_yscale('log')
        offset_line_plot(self.Jq.copy().pipe(xfilter, **filtargs),
                         x='time', y='depth', remove_mean=False,
                         offset=0, ax=ax['Jq'], **plotargs)

        # -------- T, S
        ctdargs = dict(filt=filt, filter_len=filter_len, kind=TSkind,
                    lw=0.5, t0=t0, t1=t1, add_colorbar=False)
        ax['Tplot'] = self.PlotCTD('T', ax['T'], **ctdargs)
        ax['Splot'] = self.PlotCTD('S', ax['S'], **ctdargs)

        ax['met'].set_ylabel('$τ$ (N/m²)')

        ax['N2'].legend([str(zz)+' m' for zz in self.N2.depth.values])
        ax['N2'].set_ylabel('$N²$ ($10^{-4}$)')
        limy = ax['N2'].get_ylim()
        if filt != 'bandpass':
            ax['N2'].set_ylim([0, limy[1]])

        ax['Tz'].set_ylabel('$\partial T/ \partial z$ (symlog)')
        ax['Tz'].axhline(0, color='gray', zorder=-1, linewidth=0.5)
        ax['Tz'].set_yscale('symlog', linthreshy=1e-3, linscaley=0.5)
        ax['Tz'].grid(True, axis='y', linestyle='--', linewidth=0.5)

        if 'χ' in ax:
            ax['χ'].set_title('')
            ax['χ'].set_ylabel('$χ$')
        elif 'v' in ax:
            ax['hquiv'] = self.Quiver(self.vel.time, self.vel.u, self.vel.v,
                                      ax['v'], filter_len, filt)
            ax['v'].set_title('')
            ax['v'].set_yticklabels([])
            ax['v'].set_ylabel('(u,v)')

        ax['Kt'].set_title('')
        ax['Kt'].set_ylabel('$K_T$')

        ax['Jq'].set_title('')
        ax['Jq'].axhline(0, color='gray', zorder=-1, linewidth=0.5)
        ax['Jq'].set_ylabel('$J_q^t$')
        ax['Jq'].grid(True, axis='y', linestyle='--', linewidth=0.5)

        ax['met'].set_xlim([self.χ.time.min().values,
                            self.χ.time.max().values])
        plt.gcf().autofmt_xdate()

        for name in ['N2', 'T', 'S', 'v', 'χ', 'Kt', 'Jq', 'Tz']:
            if name in ax:
                self.MarkSeasonsAndSpecials(ax[name])
                if filt == 'bandpass':
                    if (name not in ['T', 'S'] or (name in ['T', 'S']
                                             and TSkind == 'timeseries')):
                        dcpy.plots.liney(0, ax=ax[name])

        self.MarkSeasonsAndSpecials(ax['met'], season=False)

        plt.tight_layout(w_pad=2, h_pad=-0.5)

        # box = ax['T'].get_position()
        # ax['cbar'] = plt.axes([(box.x0 + box.width)*1.02,
        #                        box.y0, 0.01, box.height])
        # hcbar = plt.colorbar(hdl, cax=ax['cbar'])
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

        title = self.name + ' | ' \
                + self.GetFilterLenLabel(filt, filter_len)
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

    def Budget(self):

        from dcpy.util import MovingAverage
        import numpy as np

        pod = self.χpod[list(self.χpod.keys())[0]]

        ρ = 1025
        cp = 4200
        H = pod.depth
        T = pod.ctd1.T
        tT = pod.ctd1.time

        # rate of change of daily average temperature
        dTdt = np.diff(MovingAverage(T, 144)) / (86400)
        tavg = MovingAverage(tT, 144)
        tavg = (tavg[0:-1] + tavg[1:]) / 2

        # rate of heating of water column
        Q = ρ * cp * dTdt * H

        # budget is Q = Jq0 + Jqt
        # where both Jqs are positive when they heat the surface

        # get Jq0 on same time grid as Qavg
        Jq0 = np.interp(tavg, self.met.Jtime, self.met.Jq0)

        Jqt = Q - Jq0

        import matplotlib.pyplot as plt
        from statsmodels.nonparametric.smoothers_lowess import lowess
        import dcpy.plots

        ax1 = plt.subplot(311)
        plt.plot(MovingAverage(tT, 144), MovingAverage(T, 144))
        ax1.xaxis_date()
        plt.ylabel('T (1 day avg)')

        plt.subplot(312, sharex=ax1)
        a = lowess(dTdt, tavg, frac=0.025)
        plt.plot(a[:, 0], a[:, 1] * 86400)
        plt.ylabel('∂T/∂t (C/day)')
        dcpy.plots.liney(0)
        dcpy.plots.symyaxis()

        plt.subplot(313, sharex=ax1)
        a = lowess(Jqt.T, tavg, frac=0.025)
        plt.plot(a[:, 0], a[:, 1])
        plt.plot(tavg, Jq0)
        a = lowess(Q, tavg, frac=0.025)
        plt.plot(a[:, 0], a[:, 1])
        plt.legend(['$J_q^t$', '$J_q^0$', '$Q_{avg}$'])
        dcpy.plots.liney(0)

        dcpy.plots.symyaxis()

        plt.gcf().autofmt_xdate()
