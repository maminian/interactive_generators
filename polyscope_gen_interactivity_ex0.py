"""
This version brings in the topological feature generator
class/info from int_gen and incorporates it into the
matplotlib interactivity.

This time use a data set in 3 dimensions and try to pick out H_2 information.

This approach uses polyscope for visualization.
"""
import int_gen
#import generators_proto2
import datasets

import matplotlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import generators_proto2

import numpy as np

import polyscope

import os,pickle

pklname = 'polyscope_gen_ex2_data.pkl'

if not os.path.isfile(pklname):
    # create data
    #X = generators_proto2.noisy_circles(40, ncirc=6, eps=0.2)
    X = datasets.noisy_sphere(40,eps=0.01)

    # append a few random other things.
    X = np.vstack([X, datasets.noisy_sphere(60,eps=0.01)*1.2 - 2])

    Y = -1 + generators_proto2.noisy_circles(20)
    Y = np.hstack( [Y, 4*np.ones((Y.shape[0], 1))])

    X = np.vstack([X,Y])


    # process
    ig = int_gen.Int_gen(X, maxdim=2)
    ig.build(verbosity=1)

    # visualize

    ###############

    def assign_colors(labels, cmap=pyplot.cm.tab10, mod=20):
        # get rgb colors for each category (polyscope doesn't want alpha values)
        palette = {s:cmap(j%mod)[:3] for j,s in enumerate(np.unique(labels))}
        # assign
        return np.array([palette[s] for s in labels])
    #

    order = ig.get_order()

    #gens = {'g_%i'%i : ig.topo_features[o]['generator_ptr'] for i,o in enumerate(order[:5])}
    gens = {}
    for j,o in enumerate(order):
        flag = np.zeros(X.shape[0])
        thing = ig.topo_features[o]
        flag[thing['generator_ptr']] = 1
        colors = pyplot.cm.cividis(flag)[:,:3]
        gens[j] = {
            'name': 'H_%i, generator %i'%(thing['H_i'],j),
            'ptr':ig.topo_features[o]['generator_ptr'],
            'colors':colors,
            'H_i': thing['H_i']
            }
    #
    with open(pklname,'wb') as f:
        pickle.dump([X,gens,ig],f)
    #
else:
    with open(pklname,'rb') as f:
        dump = pickle.load(f)
    #
    X,gens,ig = dump
#

# End of computation; now visualize.

##################################################
#
# show birth/death diagrams for reference.
#
fig,ax = pyplot.subplots(1,1, figsize=(6,6))

# birth/death data
dmax = 1.05*max(ig.deaths)
ax.plot([0,dmax],[0,dmax], c=[0.5,0.5,0.5], ls='--')

ec = {hi:np.where(ig.H_i==hi)[0] for hi in range(ig.maxdim+1)}
#himap = np.zeros( len(ig.H_i), dtype=int )
himap = {}
for k,v in ec.items():
    for j,vi in enumerate(v):
        himap[j] = himap.get(j,{})
        himap[j][k] = vi
#

#scatter = ax[0].scatter(ig.births,ig.deaths, c=[default_col], picker=2)
scatters = [ax.scatter(ig.births[v], ig.deaths[v], c=[pyplot.cm.tab10(k)], picker=2, label=r'$H_{%i}$'%k) for k,v in ec.items()]

order = ig.get_order()
for j,o in enumerate(order):
    ax.text(ig.births[o], ig.deaths[o], str(j), color='k', fontsize=10)
#

ax.set_xlim([0,dmax])
ax.set_ylim([0,dmax])

fig.show()

##################################################

polyscope.init()
pc = polyscope.register_point_cloud('synth data', X)

for hi in [0,1,2]:
    cnt=0
    for j in range(len(ig.births)):
        v = gens[j]
        if cnt==4: break
        if v['H_i']!=hi:
            continue
        pc.add_color_quantity(v['name'], v['colors'])
        print(v['name'])
        cnt += 1
    #
#
polyscope.show()
