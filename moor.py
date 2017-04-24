class moor:
    ''' Class for a *single* RAMA/NRL mooring '''

    def __init__(self, lon, lat, name, datadir):

        self.datadir = datadir

        # adcp info
        self.adcp = dict()

        # ctd info
        self.temp = dict()
        self.sal = dict()
        self.dens = dict()

        # air-sea stuff
        self.τ = []
        self.τtime = []

        # chipods
        self.χpod = dict()
        self.zχpod = dict()

        # location
        self.lon = lon
        self.lat = lat

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
                                       str(name), fname, best)
        self.zχpod[name] = depth

    def Plotχpods(self, var='chi', est='best', filter_len=24*6):
        ''' Plot χ or K_T for all χpods '''

        import matplotlib.pyplot as plt
        import dcpy.util

        ax1 = plt.subplot(311)
        ax1.plot_date(self.τtime, self.τ, '-', linewidth=1)

        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313, sharex=ax1)
        labels = []

        for unit in self.χpod:
            pod = self.χpod[unit]

            if est == 'best':
                ee = pod.best

            χ = pod.chi[ee]

            ax2.plot_date(dcpy.util.datenum2datetime(χ['time']),
                          χ['N2'], '-', linewidth=1)

            pod.PlotEstimate(var, ee, hax=ax3,
                             filter_len=filter_len)
            labels.append(str(unit) + ' | ' + ee)

        ax1.set_ylabel('τ (N/m²)')

        ax2.legend(labels)
        ax2.set_ylabel('N² (s)')

        ax3.set_title('')
        ax3.set_ylabel(var)
        # plt.grid(True)

        plt.tight_layout()
