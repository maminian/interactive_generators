import ripser
import numpy as np
import datasets
from matplotlib import pyplot

rips = ripser.Rips()    # defaults are fine.

n = 100 # not really the same for all examples.

np.random.seed(0)
X1 = datasets.noisy_circle(100, eps=0.2)
bd1 = rips.fit_transform(X1)

np.random.seed(0)
X2 = datasets.kettlebell(100, eps=0.2)
bd2 = rips.fit_transform(X2)

fig,ax = pyplot.subplots(2,2, figsize=(8,8), constrained_layout=True)

fig0,ax0 = pyplot.subplots(1,2, figsize=(8,4), constrained_layout=True)
fig1,ax1 = pyplot.subplots(1,2, figsize=(8,4), constrained_layout=True)


ax[0,0].scatter(X1[:,0], X1[:,1], c='k')
ax[0,0].set_title('Noisy circle', fontsize=16)
ax0[0].scatter(X1[:,0], X1[:,1], c='k')
ax0[0].set_title('Noisy circle', fontsize=16)

ax[1,0].scatter(X2[:,0], X2[:,1], c='k')
ax[1,0].set_title('Kettlebell', fontsize=16)
ax1[0].scatter(X2[:,0], X2[:,1], c='k')
ax1[0].set_title('Kettlebell', fontsize=16)

ax0[0].axis('equal')
ax1[0].axis('equal')

ax0[0].set_xlim([-1.7,1.7])
ax0[0].set_ylim([-1.7,1.7])
ax1[0].set_xlim([-1.7,1.7])
ax1[0].set_ylim([-1.7,1.7])

# diagrams
for j,bd in enumerate([bd1,bd2]):
    axi = [ax0,ax1][j]
    for ii,hi in enumerate(bd):
        ax[j,1].scatter(hi[:,0], hi[:,1], label=r'$H_{%i}$'%ii, cmap=pyplot.cm.tab10(ii))
        axi[1].scatter(hi[:,0], hi[:,1], label=r'$H_{%i}$'%ii, cmap=pyplot.cm.tab10(ii))
    ax[j,1].axis('equal')
    ax[j,1].set_xlim([0,1.6])
    ax[j,1].set_ylim([0,1.6])

    axi[1].axis('equal')
    axi[1].set_xlim([0,1.6])
    axi[1].set_ylim([0,1.6])


    ax[j,1].set_xticks(np.arange(0,2,0.5))
    ax[j,1].set_yticks(np.arange(0,2,0.5))
    axi[1].set_xticks(np.arange(0,2,0.5))
    axi[1].set_yticks(np.arange(0,2,0.5))

    ax[j,1].plot([0,1.6],[0,1.6], c='k', ls='--')
    ax[j,1].legend(loc='lower right')
    ax[j,1].grid()
    ax[j,1].set_xlabel('Birth')
    ax[j,1].set_ylabel('Death')

    axi[1].plot([0,1.6],[0,1.6], c='k', ls='--')
    axi[1].legend(loc='lower right')
    axi[1].grid()
    axi[1].set_xlabel('Birth')
    axi[1].set_ylabel('Death')
#

fig.savefig('circ_v_kettlebell.png', dpi=120, bbox_inches='tight')
fig0.savefig('circ.png', dpi=120, bbox_inches='tight')
fig1.savefig('kettlebell.png', dpi=120, bbox_inches='tight')

fig.show()
pyplot.ion()
