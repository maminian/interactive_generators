"""
This version takes a step towards what will be done
with TDA-related stuff. Generate a few different
noisy circles; have their (x,r) pairs on the left,
and highlight the associated data on the right.
"""
import numpy as np
import matplotlib
from matplotlib import pyplot

def noisy_circ(x0=0.,y0=0.,r=1.,n=40, eps=0.1):
    import numpy as np
    th = np.linspace(0,2*np.pi,n)
    return np.vstack([
        x0 + r*np.cos(th) + eps*np.random.randn(n),
        y0 + r*np.sin(th) + eps*np.random.randn(n)
    ])
#


x0s,y0s = 4*np.random.randn(2,15)
rs = 0.5 + 3*np.random.rand(15)

default_col = [0,0,0.5,1]
highlight_col = [1,0,0,1]

circs = []
for x0,y0,r in zip(x0s,y0s,rs):
    x,y = noisy_circ(x0,y0,r)
    circs.append(
        {
            'x0': x0,
            'y0': y0,
            'r': r,
            'x': x,
            'y': y
        }
    )
#

#fig = plt.figure()
#ax = fig.add_subplot(111)
fig,ax = pyplot.subplots(1,2,
    figsize=(8,4),
    gridspec_kw={'width_ratios':[1,1]},
    sharex=True, sharey=True
)


ax[0].scatter(x0s,y0s)
for j,circ in enumerate(circs):
    sc = ax[1].scatter(circ['x'], circ['y'], c=[default_col], picker=2)
    circ['coll'] = sc
#

ax[0].set_title('(x0,y0) data; click to highlight circle')

scatter = ax[0].scatter(x0s, y0s, marker='o', c=[default_col], picker=2)

fig.tight_layout()

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

    cols = np.tile(default_col, (len(x0s),1) )
    dataind = event.ind[0]
    cols[dataind] = highlight_col
    scatter.set_color(cols)

    for j,circ in enumerate(circs):
        if j==dataind:
            circ['coll'].set_color(highlight_col)
            circ['coll'].set_zorder(10)
        else:
            circ['coll'].set_color(default_col)
            circ['coll'].set_zorder(1)
    #
#    event.artist.set_color(cols)
#    ax[1].set_ylim(-0.5, 1.5)
#    figi.show()
    fig.canvas.draw()
    return True
#    return event
#
ax[0].axis('square')
cid = fig.canvas.mpl_connect('pick_event', onpick)

fig.show()
pyplot.ion()
