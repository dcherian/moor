class moor:
    ''' Class for a *single* RAMA/NRL mooring '''

    def __init__(self, lon, lat, name, datadir):

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
        self.χpod = dict()
        self.zχpod = dict()

        # location
        self.lon = lon
        self.lat = lat

    def ReadCTD(self, fname: str, FileType: str='ramaprelim'):

        if FileType == 'ramaprelim':
            from scipy.io import loadmat
            import numpy as np

            mat = loadmat(fname, squeeze_me=True)
            self.ctd.time = mat['time'] - 367
            self.ctd.temp = mat['temp'].T
            self.ctd.sal = mat['sal'].T
            # self.ctd.dens = mat['dens']
            self.ctd.depth = mat['depth']
            self.ctd.zmat = np.tile(self.ctd.depth, (1, len(mat['time'])))

        if FileType == 'ebob':
            from scipy.io import loadmat
            import numpy as np

            mat = loadmat(self.datadir + '/ancillary/ctd/'
                          + fname + 'SP.mat', squeeze_me=True)
            temp = mat['MMT_' + fname + 'A']
            salt = mat['MMS_' + fname + 'A']
            pres = mat['MMP_' + fname + 'A']
            self.ctd.temp = np.ma.masked_array(temp,
                                               mask=np.isnan(temp))
            self.ctd.sal = np.ma.masked_array(salt,
                                              mask=np.isnan(salt))
            self.ctd.time = mat['MMTime_' + fname + 'A'] - 367
            self.ctd.zmat = np.float16(pres)
            self.ctd.depth = pres[10, :]

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
                  best: str, fname: str='Turb.mat'):

        import sys
        if 'home/deepak/python' not in sys.path:
            sys.path.append('/home/deepak/python')

        import chipy.chipy as chipy

        self.χpod[name] = chipy.chipod(self.datadir + '/data/',
                                       str(name), fname, best,
                                       depth=depth)
        self.zχpod[name] = depth

    def ChipodSeasonalSummary(self, ax=None, filter_len=86400):

        if ax is None:
            import matplotlib.pyplot as plt
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

        if len(self.χpod) > 1:
            import numpy as np
            ax.set_xticks(list(np.mean(pos, 0)))
            plt.legend((handles[0]['medians'][0],
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

    def Plotχpods(self, est: str='best', filter_len=86400):
        ''' Summary plot for all χpods '''

        import matplotlib.pyplot as plt
        from dcpy.util import smooth
        import numpy as np
        import seawater as sw

        plt.figure(figsize=[8.5, 12.5])

        # initialize axes
        nax = 8
        ax = [aa for aa in range(nax)]
        ax[0] = plt.subplot(nax, 1, 1)
        ax[0].set_title(self.name)
        if self.met.τ is not []:
            dt = (self.met.τtime[1] - self.met.τtime[0]) * 86400
            ax[0].plot_date(smooth(self.met.τtime, filter_len/dt),
                            smooth(self.met.τ, filter_len/dt), '-',
                            color='k', linewidth=0.75, zorder=1)
            limy = plt.ylim()
            ax[0].set_ylim([0, limy[1]])

        if self.met.Jq0 is not []:
            ax00 = ax[0].twinx()
            ax00.set_zorder(-1)
            dt = (self.met.Jtime[1] - self.met.Jtime[0]) * 86400
            time = smooth(self.met.Jtime, filter_len/dt)
            Jq = smooth(self.met.Jq0.copy(), filter_len/dt)
            Jq[Jq > 0] = 0
            ax00.fill_between(time, Jq, linestyle='-',
                              color='#79BEDB', linewidth=1, alpha=0.6)
            Jq = smooth(self.met.Jq0.copy(), filter_len/dt)
            Jq[Jq < 0] = 0
            ax00.fill_between(time, Jq, linestyle='-',
                              color='#e53935', linewidth=1, alpha=0.6)
            ax00.xaxis_date()
            ax00.spines['right'].set_visible(True)
            ax00.set_ylabel('$J_q^0$', labelpad=-10)

        ax[1] = plt.subplot(nax, 1, 2, sharex=ax[0])
        ax[2] = plt.subplot(nax, 1, 3, sharex=ax[0])
        ax[3] = plt.subplot2grid((nax, 1), (3, 0),
                                 rowspan=2, sharex=ax[0])

        ax[-3] = plt.subplot(nax, 1, nax-2, sharex=ax[0])  # χ
        ax[-2] = plt.subplot(nax, 1, nax-1, sharex=ax[0])  # Kt
        ax[-1] = plt.subplot(nax, 1, nax, sharex=ax[0])  # Jq
        labels = []

        xlim = [1e6, 0]
        for unit in self.χpod:
            pod = self.χpod[unit]

            xlim = [min(xlim[0], pod.time[0]),
                    max(xlim[1], pod.time[-2])]

            if est == 'best':
                ee = pod.best

            χ = pod.chi[ee]

            # i0 = np.argmin(abs(self.ctd.depth - pod.depth))
            # if self.ctd.depth[i0] > pod.depth:
            #     i0 = i0 - 1

            # dz = np.abs(self.ctd.depth[i0+1]-self.ctd.depth[i0])
            # dSdz = (self.ctd.sal[i0+1, :] - self.ctd.sal[i0, :]) / dz
            # S = (self.ctd.sal[i0+1, :] + self.ctd.sal[i0, :]) / 2
            # T = (self.ctd.temp[i0+1, :] + self.ctd.temp[i0, :]) / 2
            # alpha = np.interp(χ['time'],
            #                   self.ctd.time, sw.alpha(S, T, pod.depth))
            # beta = sw.beta(S, T, pod.depth)

            dt = (χ['time'][1] - χ['time'][0]) * 86400
            ax[1].plot_date(smooth(χ['time'], filter_len/dt),
                            smooth(χ['N2'], filter_len/dt)/1e-4,
                            '-', linewidth=0.5)
            ax[2].plot_date(smooth(χ['time'], filter_len/dt),
                            smooth(χ['dTdz'], filter_len/dt)/1e-4,
                            '-', linewidth=0.5)

            xlimtemp = ax[2].get_xlim()
            ndt = np.int(np.round(1/4/(pod.ctd1.time[1]
                                       - pod.ctd1.time[0])))
            try:
                ax[3].plot_date(pod.ctd1.time[::ndt],
                                - pod.ctd1.z[::ndt], '-',
                                linewidth=0.5, color='gray')
            except:
                ax[3].axhline(-pod.depth, color='gray',
                              linewidth=0.5)

            ax[3].set_xlim(xlimtemp)

            # dt = (self.ctd.time[1] - self.ctd.time[0]) * 86400
            # ax[3].plot_date(smooth(self.ctd.time-367, filter_len/dt),
            #                 smooth(9.81*beta*dSdz, filter_len/dt)/1e-4,
            #                 '-', linewidth=1)

            pod.PlotEstimate('chi', ee, hax=ax[-3],
                             filter_len=filter_len)
            pod.PlotEstimate('KT', ee, hax=ax[-2],
                             filter_len=filter_len)
            pod.PlotEstimate('Jq', ee, hax=ax[-1],
                             filter_len=filter_len)

            labels.append(str(pod.depth) + 'm | ' + ee)

        ax[0].set_ylabel('$τ$ (N/m²)')

        ax[1].legend(labels)
        ax[1].set_ylabel('$N²$ ($10^{-4}$)')
        limy = ax[1].get_ylim()
        ax[1].set_ylim([0, limy[1]])

        ax[2].set_ylabel('$\partial T/ \partial z$')
        ax[2].axhline(0, color='gray', zorder=-1)

        # ax[3].set_ylabel('-g β dS/dz ($10^{-4}$)')
        # ax[3].axhline(0, color='gray', zorder=-1)

        ax[-3].set_title('')
        ax[-3].set_ylabel('$χ$')

        ax[-2].set_title('')
        ax[-2].set_ylabel('$K_T$')

        ax[-1].set_title('')
        ax[-1].set_ylabel('$J_q^t$')

        plt.axes(ax[-1])
        plt.gcf().autofmt_xdate()

        from dcpy.util import MovingAverage
        dt = (self.ctd.time[1] - self.ctd.time[0])*86400
        nfilt = (86400/2)/dt
        T = MovingAverage(self.ctd.temp, nfilt, axis=0)
        S = MovingAverage(self.ctd.sal, nfilt, axis=0)
        t = MovingAverage(self.ctd.time, nfilt)
        z = MovingAverage(self.ctd.zmat, nfilt, axis=0)
        hdl = ax[3].contourf(t, -z, T.T, 20,
                             cmap=plt.get_cmap('RdYlBu_r'), zorder=-1)
        hdlS = ax[3].contour(t, -z, S.T, 6,
                             colors='gray', linewidths=0.5, zorder=-1)
        # plt.clabel(hdlS, fmt='%2.1f')

        box = ax[3].get_position()
        axColor = plt.axes([(box.x0 + box.width) * 1.03,
                            box.y0*0.95, 0.01, box.height])
        plt.colorbar(hdl, cax=axColor)
        axColor.set_ylabel('T (C)')
        ax[3].set_ylabel('depth')

        ax[0].set_xlim(xlim)

        for aa in ax[0:-1]:
            try:
                plt.setp(aa.get_xticklabels(), visible=False)
                aa.xaxis_date()
            except:
                # if T pcolor occupies two subplots
                # one of the elements in ax[] is empty.
                pass

        plt.tight_layout()

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

            ax.plot(np.concatenate([S1[mask], S2[mask]]),
                    np.concatenate([T1[mask], T2[mask]]),
                    color='k', linestyle='None', label=pod.name,
                    marker=markers[idx], alpha=0.2, zorder=2)

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
