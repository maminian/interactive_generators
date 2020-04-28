import cechmate
import numpy as np
from matplotlib import pyplot
import phat


class Int_gen:
    def __init__(self, data, maxdim=None):
        self.X = np.array(data, dtype=float)    # TODO: less data mangling 
        if maxdim is None:
            self.maxdim = max( min( data.shape[1]-1, 2 ), 0)
        else:
            self.maxdim = int(maxdim)
        #
        return
    #

    def build(self):
        '''
        Apply persistent homology using the combination of Cechmate and PHAT.

        On successful run, this function constructs:
            1. A list of dictionaries. Each dictionary corresponds to a 
                birth/death feature containing keys:
                    'pair', 'birth', 'death', 'H_i', 'generator_ptr'
                The ordering here has no significance outside of the internals 
                of the cechmate/PHAT algorithms.

            2. A collection of useful statistics in the same order for 
                visualization/access of top-level information. These are 
                flat numpy arrays for easy slicing/access without having to work 
                with dictionary/list logic. Available attributes:
                    self.births
                    self.deaths
                    self.H_i
                    self.generator_ptrs

            3. A collection of common orderings of the topological features
                based on the above information generated using by numpy.argsort.
                Available orderings:
                    self._persistence_order (death-birth; sorted from largest first)
                    self._birth_order       (sorted from smallest first)
                    self._death_order       (sorted from smallest first)

        Inputs: None; but you must instantiate the class with the data matrix first.

        Outputs: None; but the above attributes are stored in the object.
        '''

        # this part loosely follows https://cechmate.scikit-tda.org/notebooks/BasicUsage.html
        rips = cechmate.Rips(maxdim = self.maxdim)

        print('Building complex...')
        compl = rips.build(self.X)   # TODO: this is the second slowest part

        print('ordering simplices...')
        ordered_simplices = sorted(compl, key=lambda x: (x[1], len(x[0])))

        # cast as numpy array for handy array slicing later
        o_s2 = np.array(ordered_simplices)

        #
        # TODO: This is the bottleneck right now in terms of speed!
        # It's written in python; if there's a C++ version sitting around it could
        # be sped up. The python code doesn't look too crazy...
        print('Casting to sparse pivot column form...')
        columns = cechmate.solver._simplices_to_sparse_pivot_column(ordered_simplices)

        print('Building boundary matrix...')
        b_m = phat.boundary_matrix(columns=columns, representation=phat.representations.sparse_pivot_column)
        print('Computing persistence pairs...')
        pairs = b_m.compute_persistence_pairs() # boundary matrix gets reduced in-place here

    #    dgms = cechmate.solver._process_distances(pairs,ordered_simplices)

        # get largest non-infinite time/radius.
    #    dgms_cat = np.concatenate(list(dgms.values()))
    #    dgms_max = dgms_cat[np.logical_not(np.isinf(dgms_cat))].max()


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
        # REVISITED: pass on #1 until visualization/processing stage.
        # otherwise loosely following 2-4, but no visualization in this function.
        #

        pp = np.array( list(pairs) ) # cast to numpy array just to get array slicing

        # manually compute lifes; only of interest for ranking birth/death features.
#        lifes_raw = [ordered_simplices[p1][1] - ordered_simplices[p0][1] for p0,p1 in pp]
#        lifes_raw = np.array(lifes_raw)


        # 1.
        # just something to lexsort on;
        # first by homological dimension, then life of feature,
        # then (if relevant) by "death time" and "birth time"
#        summary = np.vstack([
#            pp[:,0],
#            pp[:,1],
#            -lifes_raw,
#        ])

        # Find homological dimension of each thing; following the lead of
        # cechmate.solver._process_distances().
        Hi = np.array([len(o_s2[ppi[0]][0])-1 for ppi in pp])

        # Pull out simplex information from the processed boundary matrix.
        print('Identifying generators...')
        sparse_reduced_b_m = {}
#        long_chains = {}
        for j in range(b_m._matrix.get_num_cols()):
            thing = b_m._matrix.get_col(j)
            if len(thing)>0:
                sparse_reduced_b_m[j] = np.array(thing, dtype=np.int64)
#                if len(thing)>3:
#                    long_chains[j] = np.array(thing, dtype=np.int64)
        #            print([j,thing])
        #

#        summary_order = np.lexsort(summary)
        # store for later (optional) ordering in visualization. 
        # features sorted first by death-birth, then death, then birth.
#        self._persistence_order = np.lexsort(summary)
        
        print('Putting a bow on everything...')
        topo_features = []

#        for j,ii in enumerate(summary_order):
        for ii,pair in enumerate(pp):
#            pair = pp[ii]
            idx = pair[1]
            birth = o_s2[pair[0]][1]
            death = o_s2[pair[1]][1]

            # don't bother with trivial features.
            if birth==death:
                continue
            #
            gen_simplices = o_s2[sparse_reduced_b_m[idx]][:,0]

            # vertex indices in original ordering
            g_v = np.unique(np.concatenate(gen_simplices))
            
            topo_features.append(
            {
                'pair': pair,
                'birth': birth,
                'death': death,
                'H_i': Hi[ii],
                'generator_ptr': g_v
            }
            )
        #

        # numpy array just for slicing
        self.topo_features = np.array(topo_features)

        # store summary information in flat form for easier sorting/accessing by ptr.
        self.births = np.zeros( len(self.topo_features), dtype=float)
        self.deaths = np.zeros( len(self.topo_features), dtype=float)
        self.H_i = np.zeros( len(self.topo_features), dtype=int)
        
        generator_ptrs = []

        for j,d in enumerate(self.topo_features):
            self.births[j] = d['birth']
            self.deaths[j] = d['death']
            self.H_i[j] = d['H_i']
            generator_ptrs.append( d['generator_ptr'] )
        #
        self.generator_ptrs = np.array(generator_ptrs)

        # create orderings for later use.
        self._persistence_order = np.argsort(self.births - self.deaths) # largest first
        self._birth_order = np.argsort(self.births)                     # smallest first
        self._death_order = np.argsort(self.deaths)                     # smallest first

        print('done.')
        return
    #
    def get_order(self,which='persistence'):
        '''
        Convenience function for accessing half-hidden attributes giving orderings 
        of the data.

        Inputs:
            which: string; which ordering (array of integers; a permutation) to return.
                Currently allowed values:
                    'persistence' (default; death - birth)
                    'birth'         
                    'death'
        Outputs:
            order: numpy integer array of pointers indicating a permutation to bring the 
                topological features in the corresponding order.
        '''
        if (not isinstance(which,str)) or (which not in ['persistence', 'birth', 'death']):
            raise Exception('Unexpected requested topological feature ordering "%s".'%str(which))
        #
        attr = '_%s_order'%which
        return getattr(self,attr)
    #
#
