from matplotlib import pyplot
import int_gen

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

if __name__=="__main__":

    # generate data
    X = noisy_circles(80)
    X = np.random.permutation(X)    # shuffle

    # pull in tool and run it
    ig = int_gen.Int_gen(X)
    ig.build()
    
    


    #####
    #
    # ooooweee time for a visualization
    #
    # TODO: plot each generator separately.
    #

    # pick how many of the top features to plot; their colors.
    howmany = 4
    feature_colors = [pyplot.cm.Accent(j) for j in range(howmany)]

    fig,ax = pyplot.subplots(1,2, figsize=(10,5))


    ax[0].scatter(X[:,0], X[:,1], c=[[0.8,0.8,0.8]], s=20, alpha=0.8, zorder=-100)

#    births,deaths = np.zeros( (2,len(ordered_data)), dtype=float)
#    his = np.zeros(len(ordered_data), dtype=int)
#    dim_colors = np.zeros( (len(ordered_data),4), dtype=float)
#    all_ptr = []

#    for j,od in enumerate(ordered_data):
#        births[j] = od['birth']
#        deaths[j] = od['death']
#        his[j] = od['H_i']
#        dim_colors[j] = pyplot.cm.tab10(his[j])
#        all_ptr.append( od['ptr'] )
#    #

    # last polishes
    dgms_max = 1.05*max(ig.deaths)
    ec_hi = {hi:np.where(ig.H_i==hi)[0] for hi in range(ig.maxdim +1)}

    dim_colors = np.array([pyplot.cm.tab10(hi) for hi in ig.H_i])

    for k,v in ec_hi.items():
        ax[1].scatter(ig.births[v], ig.deaths[v], c=dim_colors[v], label=r'$H_{%i}$'%k, zorder=-100)

    ax[1].plot([0,dgms_max], [0,dgms_max], c='k', linestyle='--')


    order = ig.get_order('persistence')
    for j in range(howmany):

        oj = order[j]
        od = ig.topo_features[order[j]]

        # visualize
        ax[0].scatter(X[od['generator_ptr'],0], X[od['generator_ptr'],1],
            c=[feature_colors[j]], edgecolor='k', linewidth=0.5, s=50, marker='o', label='Generator %i, dim %i'%(j,od['H_i'])
            )
        ax[1].scatter(od['birth'],od['death'],
            c=[feature_colors[j]], edgecolor='k', linewidth=0.5, s=50, marker='o'
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
