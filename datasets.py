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
