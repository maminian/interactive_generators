import numpy as np
import multiprocessing

class Bootstrapper:
    def __init__(self, X, operator, p=0.1, seed=0, nproc=1):
        self.X = X              # data; each row is a data point; self.X.shape = (n,d)
        self.op = operator      # a function to apply on subsets of X
        self.seed = seed        # seed for random number generator (default: 0)
        self.p = p              # proportion of data sampled for each bootstrap. (default: 0.1)
        self.nproc = nproc      # number of concurrent processes to use (default: 1)

        self.outputs = None     # will hold outputs of operator applied to subsets of X
        self.Pool = multiprocessing.Pool(self.nproc)

        self.s = int( max(1, np.floor( self.p * self.X.shape[0] )) )
        if self.s < 3:
            print('Warning: size of bootstrap sample %i is extremely small; consider increasing p.'%self.s)
        return
    #

    def naive_bootstrap(self, N):
        '''
        Naive bootstrap; sample a fraction p of data X with replacement.
        Pass to multiprocessing to calculate.
        '''
        if not isinstance(N,int):
            raise Exception('Expected integer number of bootstrap samples to take, got %s of type %s'%(str(N),type(N)) )
        np.random.seed( self.seed )

        subsets = [ np.random.permutation(self.X.shape[0])[:self.s] for _ in range(N) ]
        inputs = [ (i,self.X[subset]) for i,subset in enumerate(subsets) ]

        self.outputs = self.Pool.map( self.op, inputs )
        return
    #

    def init_pool(self,nproc):
        '''
        Initializes the multiprocessing Pool; just in case the user screwed up initializing the class,
        or just prefers to do it at this stage.
        '''
        assert isinstance(nproc,int)
        assert nproc>=1

        self.nproc = nproc
        self.Pool = multiprocessing.Pool(self.nproc)
        return
    #
#
