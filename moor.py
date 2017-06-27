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
        self.τ = []
        self.τtime = []

        # chipods
        self.χpod = dict()
        self.zχpod = dict()

        # location
        self.lon = lon
        self.lat = lat

    def ReadCTD(self, fname: str, FileType: str='ramaprelim'):

        if FileType == 'ramaprelim':
            from scipy.io import loadmat

            mat = loadmat(fname, squeeze_me=True)
            self.ctd.time = mat['time']
            self.ctd.temp = mat['temp']
            self.ctd.sal = mat['sal']
            # self.ctd.dens = mat['dens']
            self.ctd.depth = mat['depth']

        if FileType == 'ebob':
            from scipy.io import loadmat
            import numpy as np
            import numpy.ma

            mat = loadmat(self.datadir
                          + '/ancillary/ctd/only_temp/EBOB_'
                          + fname + '_WTMP.mat',
                          squeeze_me=True)
            temp = mat['Wtmp' + fname[-1]]
            self.ctd.temp = np.ma.masked_array(temp,
                                               mask=np.isnan(temp))
            self.ctd.time = mat['Time' + fname[-1]] - 367
            self.ctd.depth = np.float16(mat['dbar_dpth'])

            mat = loadmat(self.datadir
                          + '/ancillary/ctd/only_temp/EBOB_'
                          + fname + '_WTMP.mat',
                          squeeze_me=True)

    def ReadMet(self, fname: str, FileType: str='pmel'):

        import airsea as air
        import matplotlib.dates as dt
        import numpy as np

        if FileType == 'pmel':
            import netCDF4 as nc

            met = nc.Dataset(fname)
            spd = met['WS_401'][:].squeeze()
            z0 = abs(met['depu'][0])
            self.τtime = np.float64(met['time'][:]/24.0/60.0) \
                         + np.float64(dt.date2num(
                             dt.datetime.date(2013, 12, 1)))

        self.τ = air.windstress.stress(spd, z0)

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

    def ChipodSeasonalSummary(self, ax=None):

        if ax is None:
            import matplotlib.pyplot as plt
            plt.figure(figsize=[6.5, 4.5])
            ax = plt.gca()
            ax.set_title(self.name)

        handles = []
        labels = []
        pos = []
        for idx, name in enumerate(self.χpod):
            hdl, lbl, p = self.χpod[name].SeasonalSummary(ax=ax, idx=idx)
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

        return handles, labels, pos

    def Plotχpods(self, est: str='best', filter_len=86400):
        ''' Summary plot for all χpods '''

        import matplotlib.pyplot as plt
        from dcpy.util import smooth
        import numpy as np
        import seawater as sw

        plt.figure(figsize=[6.5, 10.5])

        # initialize axes
        nax = 8
        ax = [aa for aa in range(nax)]
        ax[0] = plt.subplot(nax, 1, 1)
        try:
            dt = (self.τtime[1] - self.τtime[0]) * 86400
            ax[0].plot_date(smooth(self.τtime, filter_len/dt),
                            smooth(self.τ, filter_len/dt), '-',
                            color='k', linewidth=1)
            limy = plt.ylim()
            ax[0].set_ylim([0, limy[1]])
        except:
            pass

        ax[1] = plt.subplot(nax, 1, 2, sharex=ax[0])
        ax[2] = plt.subplot(nax, 1, 3, sharex=ax[0])
        ax[3] = plt.subplot2grid((nax, 1), (3, 0),
                                 rowspan=2, sharex=ax[0])

        ax[-3] = plt.subplot(nax, 1, nax-2, sharex=ax[0])  # χ
        ax[-2] = plt.subplot(nax, 1, nax-1, sharex=ax[0])  # Kt
        ax[-1] = plt.subplot(nax, 1, nax, sharex=ax[0])  # Jq
        labels = []

        for unit in self.χpod:
            pod = self.χpod[unit]

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

            xlim = ax[2].get_xlim()
            ndt = np.round(1/4/(pod.ctd1.time[1]-pod.ctd1.time[0]))
            ax[3].plot_date(pod.ctd1.time[::ndt], -pod.ctd1.z[::ndt], '-',
                            linewidth=0.5, color='gray')
            ax[3].set_xlim(xlim)
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

        ax[2].set_ylabel('dT/dz')
        ax[2].axhline(0, color='gray', zorder=-1)

        # ax[3].set_ylabel('-g β dS/dz ($10^{-4}$)')
        # ax[3].axhline(0, color='gray', zorder=-1)

        import cmocean as cmo
        ndt = np.round(1/4/(self.ctd.time[1]-self.ctd.time[0]))
        ax[3].pcolormesh(self.ctd.time[::ndt], -self.ctd.depth,
                         self.ctd.temp[::ndt, :].T,
                         cmap=cmo.cm.thermal, zorder=-1)
        ax[3].set_ylabel('depth')

        ax[-3].set_title('')
        ax[-3].set_ylabel('$χ$')

        ax[-2].set_title('')
        ax[-2].set_ylabel('$K_T$')

        ax[-1].set_title('')
        ax[-1].set_ylabel('$J_q$')

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
