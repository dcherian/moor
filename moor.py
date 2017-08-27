class moor:
    ''' Class for a *single* RAMA/NRL mooring '''

    def __init__(self, lon, lat, name, datadir):

        import collections
        self.name = name
        self.datadir = datadir

        # adcp info
        self.adcp = dict()

        # ctd info
        class ctd:
            pass

        self.ctd = ctd()

        # air-sea stuff
        class met:
            pass

        self.met = met()
        self.met.τ = []  # wind stress
        self.met.τtime = []
        self.met.Jq0 = []  # net radiative flux
        self.met.Jtime = []
        self.met.P = []  # precip
        self.met.Ptime = []

        # chipods
        self.χpod = collections.OrderedDict()
        self.zχpod = collections.OrderedDict()

        # location
        self.lon = lon
        self.lat = lat

        # "special" events
        self.special = dict()

    def ReadCTD(self, fname: str, FileType: str='ramaprelim'):

        import seawater as sw
        import numpy as np
        from scipy.io import loadmat

        if FileType == 'ramaprelim':
            mat = loadmat(fname, squeeze_me=True, struct_as_record=False)
            mat = mat['rama']
            if not self.ctd.__dict__:
                # first time I'm reading CTD data
                self.ctd.time = mat.time - 367
                self.ctd.temp = mat.temp.T
                self.ctd.sal = mat.sal.T
                self.ctd.depth = mat.depth
                self.ctd.zmat = np.tile(self.ctd.depth,
                                        (len(mat.time), 1))
                self.ctd.tmat = np.tile(self.ctd.time,
                                        (len(mat.depth), 1)).T
                self.ctd.Ttmat = self.ctd.tmat
                self.ctd.Tzmat = self.ctd.zmat
                self.ctd.ρ = sw.pden(self.ctd.sal, self.ctd.temp,
                                     self.ctd.Tzmat)
            else:
                # otherwise we append
                self.ctd.time = np.concatenate([self.ctd.time,
                                                mat.time - 367])
                self.ctd.temp = np.concatenate([self.ctd.temp,
                                                mat.temp.T])
                self.ctd.sal = np.concatenate([self.ctd.sal, mat.sal.T])
                self.ctd.zmat = np.tile(self.ctd.depth,
                                        (len(self.ctd.time), 1))
                self.ctd.tmat = np.tile(self.ctd.time,
                                        (len(mat.depth), 1)).T
                self.ctd.Ttmat = self.ctd.tmat
                self.ctd.Tzmat = self.ctd.zmat
                self.ctd.ρ = sw.pden(self.ctd.sal, self.ctd.temp,
                                     self.ctd.Tzmat)

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
            mat = loadmat(self.datadir + '/ancillary/ctd/'
                          + fname + 'SP-deglitched.mat', squeeze_me=True)
            temp = mat['temp'].T
            salt = mat['salt'].T
            pres = mat['pres'].T

            self.ctd.sal = np.ma.masked_array(salt,
                                              mask=np.isnan(salt))
            self.ctd.tmat = mat['time'].T - 367
            self.ctd.zmat = np.float16(pres)
            self.ctd.time = self.ctd.tmat[:, 0]
            self.ctd.depth = pres[10, :]
            self.ctd.ρ = sw.pden(self.ctd.sal, temp, self.ctd.zmat)

            mat = loadmat(self.datadir + '/ancillary/ctd/'
                          + 'only_temp/EBOB_' + fname
                          + '_WTMP.mat', squeeze_me=True)
            temp = mat['Wtmp' + fname[-1]]
            self.ctd.temp = np.ma.masked_array(temp,
                                               mask=np.isnan(temp))
            zvec = mat['dbar_dpth']
            tvec = mat['Time' + fname[-1]] - 367
            self.ctd.Ttmat = np.tile(tvec.T, (len(zvec), 1)).T
            self.ctd.Tzmat = np.tile(zvec, (len(tvec), 1))

    def ReadMet(self, fname: str=None,
                WindType: str='pmel', FluxType: str='sat'):

        import airsea as air
        import matplotlib.dates as dt
        import numpy as np
        import netCDF4 as nc
        from scipy.interpolate import interpn

        if WindType == 'pmel':
            if fname is None:
                raise ValueError('I need a filename for PMEL met data!')

            # find RAMA met file
            import glob
            met = nc.Dataset(glob.glob(fname + '/met*.cdf')[0])
            spd = met['WS_401'][:].squeeze()
            z0 = abs(met['depu'][0])
            self.met.τtime = np.float64(met['time'][:]/24.0/60.0) \
                + np.float64(dt.date2num(dt.datetime.date(2013, 12, 1)))
            self.met.τ = air.windstress.stress(spd, z0)

        elif WindType == 'sat':
            if fname is not None:
                raise ValueError('Do not provide fname for' +
                                 ' satellite flux data!')
            met = nc.MFDataset('../tropflux/tau_tropflux*')
            lon = met['longitude'][:]
            lat = met['latitude'][:]
            time = met['time'][:]
            self.met.τ = interpn((time, lat, lon),
                                 met['tau'][:, :, :],
                                 (time, self.lat, self.lon))
            self.met.τtime = time \
                + dt.date2num(dt.datetime.date(1950, 1, 1))

        if FluxType == 'sat':
            met = nc.MFDataset('../tropflux/netflux_*')
            lon = met['longitude'][:]
            lat = met['latitude'][:]
            time = met['time'][:]
            self.met.Jq0 = interpn((time, lat, lon),
                                   met['netflux'][:, :, :],
                                   (time, self.lat, self.lon))
            self.met.Jtime = time \
                + dt.date2num(dt.datetime.date(1950, 1, 1))

            met = nc.MFDataset('../ncep/prate*')
            lat = met['lat'][:]
            lon = met['lon'][:]
            time = met['time'][:]
            self.met.P = interpn((time, np.flipud(lat), lon),
                                 np.fliplr(met['prate'][:, :, :]),
                                 (time, self.lat, self.lon))
            self.met.Ptime = time \
                + dt.date2num(dt.datetime.date(1800, 1, 1))

    def AddChipod(self, name, depth: int,
                  best: str, fname: str='Turb.mat', dir=None):

        import sys
        if 'home/deepak/python' not in sys.path:
            sys.path.append('/home/deepak/python')

        import chipy.chipy as chipy

        if dir is None:
            dir = self.datadir

        self.χpod[name] = chipy.chipod(dir + '/data/',
                                       str(name), fname, best,
                                       depth=depth)
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

    def ChipodSeasonalSummary(self, ax=None, filter_len=86400):

        import matplotlib.pyplot as plt

        if ax is None:
            plt.figure(figsize=[6.5, 4.5])
            ax = plt.gca()
            ax.set_title(self.name)

        handles = []
        labels = []
        pos = []
        for idx, name in enumerate(self.χpod):
            hdl, lbl, p = \
                    self.χpod[name].SeasonalSummary(ax=ax,
                                                    idx=idx,
                                                    filter_len=filter_len)
            handles.append(hdl)
            labels.append(lbl)
            pos.append(p)

        ax.set_title(self.name)

        if len(self.χpod) > 1:
            import numpy as np
            ax.set_xticks(list(np.mean(pos, 0)))
            ax.legend((handles[0]['medians'][0],
                       handles[-1]['medians'][0]),
                      labels)

            limy = ax.get_yticks()
            limx = ax.get_xticks()
            ax.spines['left'].set_bounds(limy[1], limy[-2])
            ax.spines['bottom'].set_bounds(limx[0], limx[-1])

        if filter_len is not None:
            ax.set_title(ax.get_title() + ' | filter_len='
                         + str(filter_len) + ' s')

        return handles, labels, pos

    def DepthPlot(self, varname, est: str='best', filter_len=86400):

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import cmocean as cmo

        s0 = 4  # min size of circles
        ns = 3  # scaling for sizea
        alpha = 0.4

        plt.figure(figsize=(6.5, 8.5))
        ax0 = plt.gca()
        ndt = np.round(1/4/(self.ctd.time[1]-self.ctd.time[0]))
        hpc = ax0.pcolormesh(self.ctd.time[::ndt], -self.ctd.depth,
                             self.ctd.temp[::ndt, :].T,
                             cmap=plt.get_cmap('RdYlBu_r'), zorder=-1)
        ax0.set_ylabel('depth')

        xlim = [1e10, -1e10]
        hscat = [1, 2]
        for idx, unit in enumerate(self.χpod):
            pod = self.χpod[unit]
            xlim[0] = np.min([xlim[0], pod.time[1]])
            xlim[1] = np.max([xlim[1], pod.time[-2]])

            var, titlestr, scale, _ = pod.ChooseVariable(varname, est)
            time, var = pod.FilterEstimate(varname,
                                           pod.time, var,
                                           filter_len=filter_len,
                                           subset=True)
            if scale == 'log':
                normvar = np.log10(var)
                dn = normvar - np.nanmin(normvar)
                size = s0 \
                    + ns * s0 * dn/np.nanstd(dn)

            # running average depths,
            # then interpolate to correct time grid
            time2, depth2 = pod.FilterEstimate('Jq', pod.ctd1.time,
                                               -pod.ctd1.z, filter_len,
                                               True)
            depth = np.interp(time, time2, depth2)

            hscat[idx] = ax0.scatter(time, depth, s=size,
                                     c=np.log10(var), alpha=alpha,
                                     cmap=cm.Greys)
            if idx == 0:
                clim = [np.nanmin(np.log10(var)),
                        np.nanmax(np.log10(var))]
            else:
                clim2 = [np.nanmin(np.log10(var)),
                         np.nanmax(np.log10(var))]
                clim = [min([clim[0], clim2[0]]),
                        max([clim[1], clim2[1]])]

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

    def avgplt(self, ax, t, x, flen, filt, **kwargs):
        from dcpy.util import MovingAverage
        from dcpy.ts import BandPassButter
        import numpy as np

        dt = (t[1]-t[0]) * 86400
        if filt == 'mean':
            t = MovingAverage(t.copy(), flen/dt)
            x = MovingAverage(x.copy(), flen/dt)

        if filt == 'bandpass':
            flen = np.array(flen.copy())
            assert(len(flen) > 1)
            x = BandPassButter(x.copy(), 1/flen, dt)

        if ax is None:
            return t, x
        else:
            hdl = ax.plot(t, x, **kwargs)
            ax.xaxis_date()
            return hdl[0]

    def Plotχpods(self, est: str='best', filt='mean', filter_len=86400):
        ''' Summary plot for all χpods '''

        import matplotlib.pyplot as plt
        from dcpy.util import MovingAverage
        import numpy as np

        plt.figure(figsize=[12.5, 6.5])
        lw = 0.5

        # initialize axes
        ax = dict()
        ax['met'] = plt.subplot(4, 2, 1)
        ax['N2'] = plt.subplot(4, 2, 3, sharex=ax['met'])
        ax['T'] = plt.subplot(4, 2, 5, sharex=ax['met'])
        ax['S'] = plt.subplot(4, 2, 7, sharex=ax['met'])
        # ax['T'] = plt.subplot2grid((4, 2), (2, 0),
        #                            rowspan=2, sharex=ax['met'])
        # ax['T'].rowNum = 3

        ax['Tz'] = plt.subplot(4, 2, 2, sharex=ax['met'])
        ax['χ'] = plt.subplot(4, 2, 4, sharex=ax['met'])
        ax['Kt'] = plt.subplot(4, 2, 6, sharex=ax['met'])
        ax['Jq'] = plt.subplot(4, 2, 8, sharex=ax['met'])

        for aa in ax:
            self.SetColorCycle(ax[aa])

        # met forcing in ax['met']
        ax['met'].set_clip_on(False)
        ax['met'].set_title(self.name + ' | '
                            + str(np.round(filter_len/86400)) + ' day '
                            + filt)

        if self.met.τ is not []:
            self.avgplt(ax['met'], self.met.τtime, self.met.τ,
                        filter_len, filt, color='k',
                        linewidth=lw, zorder=1)
            if filt == 'bandpass':
                ax['met'].set_ylim([-0.1, 0.1])
            else:
                ax['met'].set_ylim([0, 0.3])

        if self.met.Jq0 is not []:
            ax00 = ax['met'].twinx()
            ax00.set_zorder(-1)
            time, Jq = self.avgplt(None, self.met.Jtime, self.met.Jq0,
                                   filter_len, filt)
            ax00.spines['right'].set_visible(True)
            ax00.spines['left'].set_visible(False)
            self.PlotFlux(ax00, time, Jq)
            ax00.xaxis_date()
            ax00.set_ylabel('$J_q^0$ (W/m²)', labelpad=0)
            if filt == 'bandpass':
                ax00.set_ylim(np.array([-1, 1]) *
                              np.max(np.abs(ax00.get_ylim())))

        labels = []

        xlim = [1e6, 0]
        for unit in self.χpod:
            pod = self.χpod[unit]

            if est == 'best':
                ee = pod.best

            χ = pod.chi[ee]

            xlim = [min(xlim[0], pod.time[0]),
                    max(xlim[1], pod.time[-2])]

            self.avgplt(ax['N2'], χ['time'], χ['N2']/1e-4,
                        filter_len, filt, linewidth=lw)
            self.avgplt(ax['Tz'], χ['time'], χ['dTdz'],
                        filter_len, filt, linewidth=lw)

            # showing χpod depth on contourf plot
            # xlimtemp = ax['χ'].get_xlim()
            # ndt = np.int(np.round(1/4/(pod.ctd1.time[1]
            #                            - pod.ctd1.time[0])))
            # try:
            #     ax['T'].plot_date(pod.ctd1.time[::ndt],
            #                       - pod.ctd1.z[::ndt], '-',
            #                       linewidth=0.5, color='gray')
            # except:
            #     ax['T'].axhline(-pod.depth, color='gray',
            #                     linewidth=0.5)
            # ax['met'].set_xlim(xlimtemp)

            pod.PlotEstimate('chi', ee, hax=ax['χ'], filt=filt,
                             decimate=True,
                             filter_len=filter_len, linewidth=lw)
            pod.PlotEstimate('KT', ee, hax=ax['Kt'], filt=filt,
                             decimate=True,
                             filter_len=filter_len, linewidth=lw)
            pod.PlotEstimate('Jq', ee, hax=ax['Jq'], filt=filt,
                             decimate=True,
                             filter_len=filter_len, linewidth=lw)

            labels.append(str(pod.depth) + 'm')

        from cycler import cycler
        import matplotlib as mpl
        N = self.ctd.temp.shape[1]-1
        colors = mpl.cm.Greys_r(np.arange(N)/N)
        ax['T'].set_prop_cycle(cycler('color', colors))
        ax['S'].set_prop_cycle(cycler('color', colors))
        dt = np.nanmean(np.diff(self.ctd.time))*86400
        nfilt = (86400/2)/dt
        self.avgplt(ax['T'],
                    MovingAverage(self.ctd.time, nfilt, axis=0),
                    MovingAverage(self.ctd.temp[:, :-1], nfilt, axis=0),
                    flen=None, filt='None', linewidth=lw)
        self.avgplt(ax['S'],
                    MovingAverage(self.ctd.time, nfilt, axis=0),
                    MovingAverage(self.ctd.sal[:, :-1], nfilt, axis=0),
                    flen=None, filt='None', linewidth=lw)
        ax['T'].legend([str(aa)+'m' for aa in
                        np.int32(np.round(self.ctd.depth[:-1]))],
                       ncol=N)

        ax['met'].set_ylabel('$τ$ (N/m²)')

        ax['N2'].legend(set(labels))
        ax['N2'].set_ylabel('$N²$ ($10^{-3}$)')
        limy = ax['N2'].get_ylim()
        ax['N2'].set_ylim([0, limy[1]])

        ax['Tz'].set_ylabel('$\partial T/ \partial z$ (symlog)')
        ax['Tz'].axhline(0, color='gray', zorder=-1, linewidth=0.5)
        ax['Tz'].set_yscale('symlog', linthreshy=1e-3, linscaley=0.5)
        ax['Tz'].grid(True, axis='y', linestyle ='--', linewidth=0.5)

        ax['χ'].set_title('')
        ax['χ'].set_ylabel('$χ$')

        ax['Kt'].set_title('')
        ax['Kt'].set_ylabel('$K_T$')

        ax['Jq'].set_title('')
        ax['Jq'].axhline(0, color='gray', zorder=-1, linewidth=0.5)
        ax['Jq'].set_ylabel('$J_q^t$')
        ax['Jq'].grid(True, axis='y', linestyle='--', linewidth=0.5)
        # ax['Jq'].set_yscale('symlog', linthreshy=10, linscaley=2)

        ax['T'].set_title('')
        ax['T'].set_ylabel('$T$')
        ax['S'].set_title('')
        ax['S'].set_ylabel('$S$')

        ax['met'].set_xlim(xlim)
        plt.gcf().autofmt_xdate()

        # if filt == 'mean':
        #     T = MovingAverage(self.ctd.temp, nfilt, axis=0)
        #     zT = MovingAverage(self.ctd.Tzmat, nfilt, axis=0)
        #     tT = MovingAverage(self.ctd.Ttmat, nfilt, axis=0)
        #     S = MovingAverage(self.ctd.sal, nfilt, axis=0)
        #     # ρ = MovingAverage(self.ctd.ρ, nfilt, axis=0)
        #     tS = MovingAverage(self.ctd.tmat, nfilt, axis=0)
        #     zS = MovingAverage(self.ctd.zmat, nfilt, axis=0)
        #     cmap = plt.get_cmap('RdYlBu_r')
        #     colorlabel = 'T (C)'
        # else:
        #     _, T = self.avgplt(None, self.ctd.time, self.ctd.temp,
        #                        filter_len, filt)
        #     zT = self.ctd.Tzmat
        #     tT = self.ctd.Ttmat
        #     cmap = plt.get_cmap('RdBu_r')
        #     colorlabel = 'T\' (C)'
        #     _, S = self.avgplt(None, self.ctd.time, self.ctd.sal,
        #                        filter_len, filt)
        #     zS = self.ctd.zmat
        #     tS = self.ctd.tmat
        #     # _, ρ = self.avgplt(None, self.ctd.time, self.ctd.ρ,
        #     #               filter_len, filt)

        # hdl = ax[3].contourf(tS, -zS, ρ, 20, zorder=-1,
        #                     cmap=plt.get_cmap('RdYlBu_r'))
        # hdl = ax['T'].contourf(tT, -zT, T, 30, cmap=cmap, zorder=-1)
        # ax['T'].contour(tS, -zS, S, 10,
        #                 colors='k', linewidths=0.25, zorder=-1)
        # ax['T'].set_ylabel('depth')
        # ax['met'].set_xlim(xlim)

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

    def PlotAllSpectra(self, filter_len=None, nsmooth=5,
                       SubsetLength=None, ticks=None):

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6.5, 8.5))

        ax1 = plt.subplot(411)
        self.PlotSpectrum('T', filter_len=filter_len,
                          nsmooth=nsmooth,
                          SubsetLength=SubsetLength,
                          ticks=ticks, ax=ax1)
        ax1.set_title(self.name)

        ax2 = plt.subplot(412)
        self.PlotSpectrum('χ', filter_len=filter_len,
                          nsmooth=nsmooth,
                          SubsetLength=SubsetLength,
                          ticks=ticks, ax=ax2)

        ax3 = plt.subplot(413)
        self.PlotSpectrum('KT', filter_len=filter_len,
                          nsmooth=nsmooth,
                          SubsetLength=SubsetLength,
                          ticks=ticks, ax=ax3)

        ax4 = plt.subplot(414)
        self.PlotSpectrum('Jq', filter_len=filter_len,
                          nsmooth=nsmooth,
                          SubsetLength=SubsetLength,
                          ticks=ticks, ax=ax4)

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
            t, var = pod.FilterEstimate('mean', pod.time, var,
                                        filter_len=filter_len,
                                        decimate=True)

            t1, T1 = pod.FilterEstimate('mean',
                                        pod.ctd1.time, pod.ctd1.T,
                                        filter_len=filter_len, decimate=True)
            _, S1 = pod.FilterEstimate('mean',
                                       pod.ctd1.time, pod.ctd1.S,
                                       filter_len=filter_len, decimate=True)

            t2, T2 = pod.FilterEstimate('mean',
                                        pod.ctd2.time, pod.ctd2.T,
                                        filter_len=filter_len, decimate=True)
            _, S2 = pod.FilterEstimate('mean',
                                       pod.ctd2.time, pod.ctd2.S,
                                       filter_len=filter_len, decimate=True)

            # interpolate onto averaged χpod time grid
            T1 = np.interp(t, t1, T1)
            S1 = np.interp(t, t1, S1)
            T2 = np.interp(t, t2, T2)
            S2 = np.interp(t, t2, S2)

            mask = var > varmin
            frac = sum(mask)/len(mask)
            if frac < 0.1:
                alpha = 0.6
            else:
                alpha = 0.3

            ax.plot(np.concatenate([S1[mask], S2[mask]]),
                    np.concatenate([T1[mask], T2[mask]]),
                    color='k', linestyle='None', label=pod.name,
                    marker=markers[idx], alpha=alpha, zorder=2)

        for index, z in enumerate(self.ctd.depth):
            S = self.ctd.sal[:, index]
            T = self.ctd.temp[:, index]
            ax.scatter(S[::10], T[::10], s=size,
                       facecolor=colors[index], alpha=0.1,
                       label='CTD '+str(np.int(z))+' m', zorder=-1)

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
        cs = ax.contour(Smat, Tmat, ρ, colors='gray',
                        linestyles='dashed', zorder=-1)
        ax.clabel(cs, fmt='%.1f')

        # labels
        ax.set_xlabel('S')
        ax.set_ylabel('T')
        ax.set_title(self.name)

        ax.annotate(name + '$^{' +
                    str(np.round(filter_len/86400, decimals=1))
                    + ' d}$ > ' + str(varmin),
                    xy=[0.75, 0.9], xycoords='axes fraction',
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
        ax[1].set_xticklabels([''] + [str('{0:.1f}').format(1/aa)
                                      for aa in ax[1].get_xticks()[1:]])

    def ExtractTimeRange(self, t1, v1, t2, v2, ndayavg=None, season=None):

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
            dt1 = (t1[2]-t1[1])*86400
            v1 = MovingAverage(v1, ndayavg * 86400/dt1)
            t1 = MovingAverage(t1, ndayavg * 86400/dt1)

            dt2 = (t2[2]-t2[1])*86400
            v2 = MovingAverage(v2, ndayavg * 86400/dt2)
            t2 = MovingAverage(t2, ndayavg * 86400/dt2)

        # extract season if asked to
        if season is not None:
            from dcpy.util import ExtractSeason
            t1, v1 = ExtractSeason(t1, v1, season)
            t2, v2 = ExtractSeason(t2, v2, season)

        v2i = np.interp(t1, t2, v2)

        return t1, v1, v2i

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

        title = self.name
        if filt is not None:
            title += ' | ' + str(filter_len/86400) \
                     + ' ' + filt
        else:
            title += ' | unfiltered'
        ax0.set_title(title)

        for metidx, metvar in enumerate(metvars):
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
                met = self.met.τ
                tmet = self.met.τtime
                axes = [ax4, ax5]
                label = '$1000τ$'
                self.avgplt(ax0, tmet, met*1000, filter_len,
                            filt, label=label)
                dcpy.ts.PlotSpectrum(10*met, nsmooth=20, ax=ax1)

            for idx, unit in enumerate(self.χpod):
                pod = self.χpod[unit]

                v2 = pod.Jq[pod.best].copy()
                t1, v1, v2i = self.ExtractTimeRange(tmet.copy(), met.copy(),
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
                dcpy.plots.FillRectangle(86400/filter_len, ax=axes[0])
                dcpy.plots.FillRectangle(86400/filter_len, ax=axes[1])
            if fbands is not None:
                dcpy.plots.linex(fbands, ax=axes)

        ax0.legend(ncol=2)
        dcpy.plots.liney(0, ax0, linestyle='-')
        ax0.set_xticks(ax0.get_xticks()[::2])

        ax1.set_ylabel('PSD')
        if filter_len is not None:
            dcpy.plots.FillRectangle(86400/filter_len, ax=ax1)
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
        dTdt = np.diff(MovingAverage(T, 144))/(86400)
        tavg = MovingAverage(tT, 144)
        tavg = (tavg[0:-1] + tavg[1:])/2

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
        plt.plot(a[:,0], a[:,1] * 86400)
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
