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

    def ReadMet(self, fname, FileType='pmel'):

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

    def AddChipod(self, name, fname, depth, best):

        import sys
        if 'home/deepak/python' not in sys.path:
            sys.path.append('/home/deepak/python')

        import chipy.chipy as chipy

        self.χpod[name] = chipy.chipod(self.datadir + '/data/',
                                       str(name), fname, best,
                                       depth=depth)
        self.zχpod[name] = depth

    def Plotχpods(self, var: str='chi', est: str='best', filter_len=24*6):
        ''' Plot χ or K_T for all χpods '''

        import matplotlib.pyplot as plt
        from dcpy.util import smooth
        import matplotlib.dates as dt
        import numpy as np

        plt.figure(figsize=[6.5, 8.5])

        ax1 = plt.subplot(511)
        ax1.plot_date(smooth(self.τtime, 24*6),
                      smooth(self.τ, 24*6), '-',
                      color='k', linewidth=1)
        limy = plt.ylim()
        ax1.set_ylim([0, limy[1]])

        ax2 = plt.subplot(512, sharex=ax1)
        ax3 = plt.subplot(513, sharex=ax1)
        ax4 = plt.subplot(514, sharey=ax3, sharex=ax1)
        ax5 = plt.subplot(515, sharex=ax1)
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

            ax2.plot_date(smooth(dt.date2num(χ['time']), 24*6),
                          smooth(χ['N2'], 24*6)/1e-4,
                          '-', linewidth=1)
            ax3.plot_date(smooth(dt.date2num(χ['time']), 24*6),
                          smooth(9.81*3e-4*χ['dTdz'], 24*6)/1e-4,
                          '-', linewidth=1)
            ax4.plot_date(smooth(self.ctd.time-367, 24*6),
                          smooth(9.81*7.6e-4*dSdz, 24*6)/1e-4,
                          '-', linewidth=1)

            pod.PlotEstimate(var, ee, hax=ax5,
                             filter_len=filter_len)
            labels.append(str(pod.depth) + 'm | ' + ee)

        ax1.set_ylabel('τ (N/m²)')

        ax2.legend(labels)
        ax2.set_ylabel('N² (1e-4 s)')
        limy = ax2.get_ylim()
        ax2.set_ylim([0, limy[1]])

        ax3.set_ylabel('g α dT/dz (1e-4)')
        ax3.axhline(0, color='gray', zorder=-1)

        ax4.set_ylabel('-g β dS/dz (1e-4)')
        ax4.axhline(0, color='gray', zorder=-1)

        ax5.set_title('')
        ax5.set_ylabel(var)
        # plt.grid(True)

        for aa in [ax1, ax2, ax3, ax4]:
            plt.setp(aa.get_xticklabels(), visible=False)

        plt.tight_layout()

        return [ax1, ax2, ax3, ax4, ax5]
