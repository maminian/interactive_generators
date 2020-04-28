"""
This version brings in the topological feature generator 
class/info from int_gen and incorporates it into the 
matplotlib interactivity. 

Use the dataset in generators_proto2.
"""
import int_gen
import generators_proto2

import matplotlib
from matplotlib import pyplot

import numpy as np

#params
default_col = [0,0,0.5,1]
highlight_col = [1,0,0,1]

# create data
X = generators_proto2.noisy_circles(40, ncirc=6, eps=0.2)

# process
ig = int_gen.Int_gen(X)
ig.build(verbosity=1)

# visualize

fig,ax = pyplot.subplots(1,2,
    figsize=(8,4),
    gridspec_kw={'width_ratios':[1,1]}
)

# birth/death data
dmax = 1.05*max(ig.deaths)
ax[0].plot([0,dmax],[0,dmax], c=[0.5,0.5,0.5], ls='--')

ec = {hi:np.where(ig.H_i==hi)[0] for hi in range(ig.maxdim+1)}
#himap = np.zeros( len(ig.H_i), dtype=int )
himap = {}
for k,v in ec.items():
    for j,vi in enumerate(v):
        himap[j] = himap.get(j,{})
        himap[j][k] = vi
#

#scatter = ax[0].scatter(ig.births,ig.deaths, c=[default_col], picker=2)
scatters = [ax[0].scatter(ig.births[v], ig.deaths[v], c=[pyplot.cm.tab10(k)], picker=2, label=r'$H_{%i}$'%k) for k,v in ec.items()]

ax[0].set_xlim([0,dmax])
ax[0].set_ylim([0,dmax])

# junk to aid visualization
global highlight_flag
global mysc
highlight_flag = False
mysc = None


# original dataset
data_scatter = ax[1].scatter(X[:,0], X[:,1], c=[default_col])


def highlight(ii,axi):
    '''
    highlight a (birth,death) by putting a single scatter pt there.
    pyplot's scatter object returned from function to remove later.
    Uses information from "scatter" defined above. Temp code.
    '''
    s0 = scatters[0].get_sizes()
    s0 = s0[0] if len(s0)==1 else s0[ii]
    s1 = 4*s0

    x,y = ig.births[ii], ig.deaths[ii]

    sc = axi.scatter([x,x],[y,y], 
        c=[[255./255,235./255,161./255],[1,0,0]], s=[s1,s0], edgecolor=[[0,0,0,1],[0,0,0,0]], lw=2, zorder=100)
    return sc
#

def onpick(event):
    global highlight_flag
    global mysc

#    if event.artist!=scatter: 
    which_scatter = [event.artist==s for s in scatters]
    if not any(which_scatter):
        if highlight_flag:
            mysc.remove()
            highlight_flag = False
        return True

    which_scatter = np.where(which_scatter)[0][-1]
    scatter = scatters[which_scatter]

    N = len(event.ind)
    if not N: return True

    # proceed normally
#    for l in ax[1].lines[::-1]:
#        l.remove()
#    for e in ax[1].texts[::-1]:
#        e.remove()

#    bd_colors = np.tile(default_col, (len(ig.births),1) )
    data_colors = np.tile(default_col, (len(X),1) )

    dataind = event.ind[0]


    dataind = himap[dataind][which_scatter]
#    bd_colors[dataind] = highlight_col
#    scatter.set_color(bd_colors)
    if highlight_flag:
        mysc.remove()
        highlight_flag = False
    if not highlight_flag:
        mysc = highlight(dataind, ax[0])
        highlight_flag = True
    #

    data_colors[ig.topo_features[dataind]['generator_ptr']] = highlight_col

    data_scatter.set_color(data_colors)

    fig.canvas.draw()
    return True
#

ax[0].set_title('(birth,death) data; click to highlight generator')
ax[1].set_title('data')
ax[0].legend(loc='lower right')
ax[0].set_xlabel('birth')
ax[0].set_ylabel('death')

ax[0].axis('square')
ax[1].axis('equal')

cid = fig.canvas.mpl_connect('pick_event', onpick)

fig.tight_layout()
fig.show()
pyplot.ion()
