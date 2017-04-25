class moor:
    ''' Class for a *single* RAMA/NRL mooring '''

    def __init__(self, lon, lat, name, datadir):

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

    def ReadMet(self, fname: str, FileType: str='pmel'):

        import airsea as air
        import matplotlib.dates as dt

        if FileType == 'pmel':
            import netCDF4 as nc

            met = nc.Dataset(fname)
            spd = met['WS_401'][:].squeeze()
            z0 = abs(met['depu'][0])
            self.τtime = met['time'][:]/24/60 \
                         + dt.date2num(
                             dt.datetime.date(2013, 12, 1))

        self.τ = air.windstress.stress(spd, z0)

    def AddChipod(self, name: str, fname: str, depth: int, best: str):

        import sys
        if 'home/deepak/python' not in sys.path:
            sys.path.append('/home/deepak/python')

        import chipy.chipy as chipy

        self.χpod[name] = chipy.chipod(self.datadir + '/data/',
                                       str(name), fname, best,
                                       depth=depth)
        self.zχpod[name] = depth

    def Plotχpods(self, est: str='best', filter_len=24*6):
        ''' Summary plot for all χpods '''

        import matplotlib.pyplot as plt
        from dcpy.util import smooth
        import numpy as np
        import seawater as sw

        plt.figure(figsize=[6.5, 9.5])

        # initialize axes
        nax = 7
        ax = [aa for aa in range(nax)]
        ax[0] = plt.subplot(nax, 1, 1)
        ax[0].plot_date(smooth(self.τtime, 24*6),
                        smooth(self.τ, 24*6), '-',
                        color='k', linewidth=1)
        limy = plt.ylim()
        ax[0].set_ylim([0, limy[1]])

        ax[1] = plt.subplot(nax, 1, 2, sharex=ax[0])
        ax[2] = plt.subplot(nax, 1, 3, sharex=ax[0])
        ax[3] = plt.subplot(nax, 1, 4, sharey=ax[2], sharex=ax[0])

        ax[-3] = plt.subplot(nax, 1, nax-2, sharex=ax[0])  # χ
        ax[-2] = plt.subplot(nax, 1, nax-1, sharex=ax[0])  # Kt
        ax[-1] = plt.subplot(nax, 1, nax, sharex=ax[0])  # Jq
        labels = []

        for unit in self.χpod:
            pod = self.χpod[unit]

            if est == 'best':
                ee = pod.best

            χ = pod.chi[ee]

            i0 = np.argmin(abs(self.ctd.depth - pod.depth))
            if self.ctd.depth[i0] > pod.depth:
                i0 = i0 - 1

            dz = np.abs(self.ctd.depth[i0+1]-self.ctd.depth[i0])
            dSdz = (self.ctd.sal[i0+1, :] - self.ctd.sal[i0, :]) / dz
            S = (self.ctd.sal[i0+1, :] + self.ctd.sal[i0, :]) / 2
            T = (self.ctd.temp[i0+1, :] + self.ctd.temp[i0, :]) / 2
            alpha = np.interp(χ['time'],
                              self.ctd.time, sw.alpha(S, T, pod.depth))
            beta = sw.beta(S, T, pod.depth)

            ax[1].plot_date(smooth(χ['time'], 24*6),
                            smooth(χ['N2'], 24*6)/1e-4,
                            '-', linewidth=1)
            ax[2].plot_date(smooth(χ['time'], 24*6),
                            smooth(9.81*alpha*χ['dTdz'], 24*6)/1e-4,
                            '-', linewidth=1)
            ax[3].plot_date(smooth(self.ctd.time-367, 24*6),
                            smooth(9.81*beta*dSdz, 24*6)/1e-4,
                            '-', linewidth=1)

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

        ax[2].set_ylabel('g α dT/dz ($10^{-4}$)')
        ax[2].axhline(0, color='gray', zorder=-1)

        ax[3].set_ylabel('-g β dS/dz ($10^{-4}$)')
        ax[3].axhline(0, color='gray', zorder=-1)

        ax[-3].set_title('')
        ax[-3].set_ylabel('$χ$')

        ax[-2].set_title('')
        ax[-2].set_ylabel('$K_T$')

        ax[-1].set_title('')
        ax[-1].set_ylabel('$J_q$')

        for aa in ax[0:-1]:
            plt.setp(aa.get_xticklabels(), visible=False)

        plt.tight_layout()

        return ax
