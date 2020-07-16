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
