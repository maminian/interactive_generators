"""
compute the mean and stddev of 100 data sets and plot mean vs stddev.
When you click on one of the mu, sigma points, plot the raw data from
the dataset that generated the mean and stddev

via https://matplotlib.org/3.1.1/users/event_handling.html#object-picking

I'm slowly adapting this to my needs...
first step is to have two fixed axes and do the
plotting in the right axis in a fixed way on the
clicky event.
"""
import numpy as np
import matplotlib
from matplotlib import pyplot

X = np.random.rand(100, 1000)
xs = np.mean(X, axis=1)
ys = np.std(X, axis=1)

default_col = [0,0,0.5,1]
highlight_col = [1,0,0,1]

#fig = plt.figure()
#ax = fig.add_subplot(111)
fig,ax = pyplot.subplots(1,2,
    figsize=(12,4),
    gridspec_kw={'width_ratios':[1,2]}
)

ax[0].set_title('click on point to plot time series')

scatter = ax[0].scatter(xs, ys, marker='o', c=[default_col], picker=2)

ax[1].set_ylim(-0.5, 1.5)

prev = np.nan


def onpick(event):
    if event.artist!=scatter: return True
#    if not isinstance(event.artist, matplotlib.collections.PathCollection):
#        return True

#    print(event.ind)
    N = len(event.ind)
    if not N: return True

    # proceed normally
    for l in ax[1].lines[::-1]:
        l.remove()
    for e in ax[1].texts[::-1]:
        e.remove()

    cols = np.tile(default_col, (len(xs),1) )
#    event.artist.set_color(highlight_col)
#    figi = plt.figure()
    for subplotnum, dataind in enumerate(event.ind[:1]):
#        print(subplotnum)
#        print(dataind)
#        ax = figi.add_subplot(N,1,subplotnum+1)
        cols[dataind] = highlight_col
        ax[1].plot(X[dataind], c=pyplot.cm.tab10(0))
        ax[1].text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f'%(xs[dataind], ys[dataind]),
                transform=ax[1].transAxes, va='top')
    #
    event.artist.set_color(cols)
    ax[1].set_ylim(-0.5, 1.5)
#    figi.show()
    fig.canvas.draw()
    return True
#    return event
#

cid = fig.canvas.mpl_connect('pick_event', onpick)

fig.show()
pyplot.ion()
