import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def _colorbar(hdl, ax=None, format='%.2f'):

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
    hcbar = plt.colorbar(hdl, axcbar,
                         format=mpl.ticker.FormatStrFormatter(format),
                         ticks=mpl.ticker.MaxNLocator('auto'))

    return hcbar
