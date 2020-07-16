def noisy_sphere(n,eps=0.05):
    import numpy as np
    # uniform on sphere via Marsaglia's method
    data = np.zeros((n,3), dtype=float)
    i=0
    while i<n:
        x1,x2 = -1+2*np.random.rand(2)
        s = x1**2 + x2**2
        while s >=1:
            x1,x2 = -1+2*np.random.rand(2)
            s = x1**2 + x2**2
        data[i] = [
            2*x1*np.sqrt(1-s) + eps*np.random.randn(),
            2*x2*np.sqrt(1-s) + eps*np.random.randn(),
            1-2*s + eps*np.random.randn()
        ]
        i += 1
    #
    return data
#

def noisy_circle(n, x0=0., y0=0., eps=0.1, r=1.):
    import numpy as np
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
    import numpy as np
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
def kettlebell(n, eps=0.1):
    import numpy as np
    circ = noisy_circle(n,0,0,eps=eps,r=1)
    bottom = np.random.rand(n,2)
    bottom[:,1] *= 0.8
    bottom[:,1] -= 1.6
    bottom[:,0] *= 1.5
    bottom[:,0] -= 0.75
    data = np.vstack([circ,bottom])
    return data
#
