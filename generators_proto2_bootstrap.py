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
    if True:
        centers = 5*np.random.randn(ncirc,2)
        radii = 0.5 + 3*np.random.rand(ncirc)
    else:
        centers = np.array([
            [-1,-1],
            [3,0],
            [0,2]
            ])
        radii = [2.5,1.5,1]
    #
    return np.vstack([
        noisy_circle(n, ci[0], ci[1], eps=eps, r=ri) for ci,ri in zip(centers,radii)
    ])
#

def noisy_circle_plus_background(n):
    X = noisy_circle(n)

    xl,yl = np.min(X, axis=0)
    xr,yr = np.max(X, axis=0)
    n_small = max(1, int(0.2*n))

    Xnoise = np.zeros( (n_small, X.shape[1]) )
    Xnoise[:,0] = xl + (xr-xl)*np.random.rand(n_small)
    Xnoise[:,1] = yl + (yr-yl)*np.random.rand(n_small)

    X = np.vstack([X,Xnoise])
    return X
#

def calculate_ig(inputs):
    # to be used with multiprocessing.
    #X = noisy_circles(80)
    #X = np.random.permutation(X)    # shuffle

    # pull in tool and run it
    import int_gen
    todisplay,data = inputs

    ig = int_gen.Int_gen(data)
    ig.build(verbosity=0)
    print(todisplay)

    return ig
#

def share_lims(ax1,ax2):
    xl1,xr1 = ax1.get_xlim()
    xl2,xr2 = ax2.get_xlim()
    xl = min(xl1,xl2)
    xr = max(xr1,xr2)

    yl1,yr1 = ax1.get_xlim()
    yl2,yr2 = ax2.get_xlim()
    yl = min(yl1,yl2)
    yr = max(yr1,yr2)

    ax1.set_xlim([xl,xr])
    ax2.set_xlim([xl,xr])
    ax1.set_ylim([yl,yr])
    ax2.set_ylim([yl,yr])
    return
#

if __name__=="__main__":
    from matplotlib import pyplot,rcParams
    import bootstrap

    pyplot.style.use('dark_background')
    rcParams['font.size'] = 12


#    X = noisy_circles(200,5)
    X = noisy_circle_plus_background(300)
    bs = bootstrap.Bootstrapper(X, calculate_ig, p=0.1, nproc=4)
    bs.naive_bootstrap(1000)

    all_births = np.concatenate( [igi.births for igi in bs.outputs] )
    all_deaths = np.concatenate( [igi.deaths for igi in bs.outputs] )
    br = max(all_births)
    pr = max(all_deaths-all_births)

    plotobjs = []
    fig,ax = pyplot.subplots(1,3, figsize=(13,4), constrained_layout=True)
    obj = ax[0].scatter(X[:,0], X[:,1], s=50)
    plotobjs = [obj]
    for j,mycm in zip([1,2], [pyplot.cm.cividis, pyplot.cm.magma]):
        masks = [igi.H_i==(j-1) for igi in bs.outputs]
        all_b = np.concatenate( [igi.births[mask] for igi,mask in zip(bs.outputs,masks)] )
        all_d = np.concatenate( [igi.deaths[mask] for igi,mask in zip(bs.outputs,masks)] )
        obj = ax[j].hexbin(all_b,all_d - all_b, cmap=mycm, mincnt=1, edgecolors=None, gridsize=20, extent=(0,max(br,pr),0,max(br,pr)))

        ax[j].text(0.05,0.95, r'$PH_{%i}$'%(j-1), ha='left', va='top', transform=ax[j].transAxes)

        bincnts = obj.get_array()
        obj.set_clim(0, int( np.quantile(bincnts,0.9) ) )
        plotobjs.append(obj)
    #

    # plot decorations
    ax[1].set_ylabel(r'$d_i - b_i$')
    ax[1].set_xlabel(r'$b_i$')
    ax[2].set_xlabel(r'$b_i$')
    ax[2].set_yticklabels([])

    cbar = fig.colorbar(plotobjs[-1],ax=ax[1:], fraction=0.001, pad=0.01,extend='max')
#    ax[2].get_shared_x_axes().join(ax[2], ax[1])
#    ax[2].get_shared_y_axes().join(ax[2], ax[1])

#    ax[2].axis('equal')
    fig.show()

    pyplot.ion()
