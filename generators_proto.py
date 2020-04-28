import cechmate
from matplotlib import pyplot
import phat


import numpy as np
#

np.random.seed(2718281828)

def noisy_circle(n, x0=0., y0=0., eps=0.1, r=1.):
    th = np.linspace(0,2*np.pi,n)
    return np.vstack([
        x0 + r*np.cos(th) + eps*np.random.randn(n),
        y0 + r*np.sin(th) + eps*np.random.randn(n)
    ]).T
#
def noisy_circles(n,ncirc=2, eps=0.1):
    '''
    Places 3 circles, each with n points. n*3 total points.
    ncirc argument ignored.
    '''
    #centers = 5*np.random.randn(ncirc,2)
    centers = np.array([
        [-1,-1],
        [3,0],
        [0,2]
        ])
    radii = [2.5,1.5,1]
    return np.vstack([
        noisy_circle(n, ci[0], ci[1], eps=eps, r=ri) for ci,ri in zip(centers,radii)
    ])
#


X = noisy_circles(80, 3)
X = np.random.permutation(X)    # shuffle

# this part loosely follows https://cechmate.scikit-tda.org/notebooks/BasicUsage.html
rips = cechmate.Rips(maxdim=1)
compl = rips.build(X)   # this is the second slowest part
ordered_simplices = sorted(compl, key=lambda x: (x[1], len(x[0])))

# cast as numpy array for handy array slicing later
o_s2 = np.array(ordered_simplices)

#
# This is the bottleneck right now in terms of speed!
# It's written in python; if there's a C++ version sitting around it could
# be sped up. The python code doesn't look too crazy...
columns = cechmate.solver._simplices_to_sparse_pivot_column(ordered_simplices)

b_m = phat.boundary_matrix(columns=columns, representation=phat.representations.sparse_pivot_column)
pairs = b_m.compute_persistence_pairs() # boundary matrix gets reduced in-place here

dgms = cechmate.solver._process_distances(pairs,ordered_simplices)

# get largest non-infinite time/radius.
dgms_cat = np.concatenate(list(dgms.values()))
dgms_max = dgms_cat[np.logical_not(np.isinf(dgms_cat))].max()


#
# OK, here's the sketch of what we're doing
# to get out generators:
#
# 1. Use some criterion to identify which birth/death pairs you want
#    (e.g. lexsort by homological dimension, then lifetime)
# 2. Identify the "pairs" in the associated table (from b_m.compute_persistence_pairs)
# 3. Associate "pairs" (boundary matrix columns) with associated simplices
#    by identifying nonzero entries (b_m.get_column(j)); these are indexes
#    for ordered_simplices)
# 4. Visualize/attach information however you like.
#
pp = np.array( list(pairs) ) # cast to numpy array just to get array slicing

# manually compute lifes; only of interest for ranking birth/death features.
lifes_raw = [ordered_simplices[p1][1] - ordered_simplices[p0][1] for p0,p1 in pp]
lifes_raw = np.array(lifes_raw)


# 1.
# just something to lexsort on;
# first by homological dimension, then life of feature,
# then (if relevant) by "death time" and "birth time"
summary = np.vstack([
    pp[:,0],
    pp[:,1],
    -lifes_raw,
])

# Find homological dimension of each thing; following the lead of
# cechmate.solver._process_distances().
Hi = np.array([len(o_s2[ppi[0]][0])-1 for ppi in pp])

# Pull out simplex information from the processed boundary matrix.
sparse_reduced_b_m = {}
long_chains = {}
for j in range(b_m._matrix.get_num_cols()):
    thing = b_m._matrix.get_col(j)
    if len(thing)>0:
        sparse_reduced_b_m[j] = np.array(thing, dtype=np.int64)
        if len(thing)>3:
            long_chains[j] = np.array(thing, dtype=np.int64)
#            print([j,thing])
#

summary_order = np.lexsort(summary)

ordered_data = []

for j,ii in enumerate(summary_order):
    pair = pp[ii]
    idx = pair[1]
    birth = o_s2[pair[0]][1]
    death = o_s2[pair[1]][1]

    # don't bother with trivial features.
    if birth==death:
        continue
    gen_simplices = o_s2[sparse_reduced_b_m[idx]][:,0]

    # vertex indices in original ordering
    g_v = np.unique(np.concatenate(gen_simplices))
    
    ordered_data.append(
    {
        'pair': pair,
        'birth': birth,
        'death': death,
        'H_i': Hi[ii],
        'ptr': g_v
    }
    )
#


if __name__=="__main__":

    #####
    #
    # ooooweee time for a visualization
    #
    # TODO: plot each generator separately.
    #

    # pick how many of the top features to plot; their colors.
    howmany = 4
    colors = [pyplot.cm.Accent(j) for j in range(howmany)]

    fig,ax = pyplot.subplots(1,2, figsize=(10,5))


    ax[0].scatter(X[:,0], X[:,1], c=[[0.8,0.8,0.8]], s=20, alpha=0.8, zorder=-100)

    births,deaths = np.zeros( (2,len(ordered_data)), dtype=float)
    his = np.zeros(len(ordered_data), dtype=int)
    dim_colors = np.zeros( (len(ordered_data),4), dtype=float)
    all_ptr = []

    for j,od in enumerate(ordered_data):
        births[j] = od['birth']
        deaths[j] = od['death']
        his[j] = od['H_i']
        dim_colors[j] = pyplot.cm.tab10(his[j])
        all_ptr.append( od['ptr'] )
    #

    # last polishes
    dgms_max = 1.05*max(deaths)
    ec_hi = {hi:np.where(his==hi)[0] for hi in np.unique(his)}

    for k,v in ec_hi.items():
        ax[1].scatter(births[v],deaths[v],c=dim_colors[v], label=r'$H_{%i}$'%k, zorder=-100)

    ax[1].plot([0,dgms_max], [0,dgms_max], c='k', linestyle='--')


    for j in range(howmany):

        od = ordered_data[j]

        # visualize
        ax[0].scatter(X[od['ptr'],0], X[od['ptr'],1],
            c=[colors[j]], edgecolor='k', linewidth=0.5, s=50, marker='o', label='Generator %i, dim %i'%(j,od['H_i'])
            )
        ax[1].scatter(od['birth'],od['death'],
            c=[colors[j]], edgecolor='k', linewidth=0.5, s=50, marker='o'
            )

        fudge=0.1   # get it off the point...
        ax[1].text(od['birth']+fudge,od['death'], 'Generator %i, dim %i'%(j, od['H_i']))
    #

    ax[0].legend(loc='best')
    ax[1].legend(loc='best')

    ax[0].axis('equal')
    ax[1].axis('equal')


    fig.tight_layout()
    fig.show()
    pyplot.ion()
