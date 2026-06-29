'''
Module implementing class to handle datasets and related settings.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from SDP_miRNA import utils
import pandas as pd
import numpy as np
import tqdm
import scipy

# ------------------------------------------------
# Dataset class
# ------------------------------------------------

class Dataset():
    def __init__(self):
        '''Initialise dataset settings.'''

        # data
        self.sparse_A = None
        self.sparse_B = None

        # selection
        self.gene_queries = None

        # size
        self.cells = None
        self.total_gene_queries = None

        # capture efficiency
        self.beta = None

        # bootstrap settings
        self.d = None
        self.resamples = None
        self.confidence = None
        self.query_chunk_size = None
        self.bootstrap_chunk_size = None

        # moment bounds
        self.moment_bounds = None
    
    def construct_dataset(self, adata_A, adata_B, beta, gene_queries=None):
        '''Setup object given data.'''

        # store objects
        self.sparse_A = adata_A.X
        self.sparse_B = adata_B.X
        self.beta = beta

        # default selection of 1st A gene paired with all B genes
        if gene_queries is None:
            gene_queries = [[[0], [i]] for i in range(adata_B.n_vars)]
        self.gene_queries = gene_queries

        # size
        self.cells = adata_A.n_obs
        self.total_gene_queries = len(self.gene_queries)

    def bootstrap(self, d, confidence=0.95, resamples=1000, query_chunk_size=100, bootstrap_chunk_size=100, tqdm_disable=False):
        '''
        Compute bootstrap percentile confidence intervals on moments over the
        species in each gene query up to order d

        Arguments:
            self: dataset object containing count information
            d: maximum moment order
            confidence: percentile confidence level
            resamples: number of bootstrap resamples
            query_chunk_size: division of gene queries for memory safety
            bootstrap_chunk_size: division of bootstrap resamples for memory safety
            tqdm_disable: bool to hide progress bar

        Returns:
            moment_bounds: all confidence intervals stored as attribute
        '''

        # helpful values
        queries_A_list = [query[0] for query in self.gene_queries]
        queries_B_list = [query[1] for query in self.gene_queries]
        S_A = len(queries_A_list[0])
        S_B = len(queries_B_list[0])
        S = S_A + S_B
        alpha = 1 - confidence

        # store settings
        self.d = d
        self.confidence = confidence
        self.resamples = resamples
        self.query_chunk_size = query_chunk_size
        self.bootstrap_chunk_size = bootstrap_chunk_size
        self.S = S
        self.powers = utils.compute_powers(S, d)
        self.Nd = utils.compute_Nd(S, d)

        # shorthand
        N = self.cells

        # initialize random generator
        rng = np.random.default_rng()
    
        # Compute array of moments needed
        exponent_grid = np.array(self.powers, dtype=np.int32)
        
        # convert to CSR: better multiple column indexing via transpose
        A_csr = scipy.sparse.csr_matrix(self.sparse_A)
        B_csr = scipy.sparse.csr_matrix(self.sparse_B)
        
        # Bootstrap resamples indices
        boot_indices = rng.integers(0, N, size=(N, resamples))
        
        # Store final confidence intervals
        moment_intervals = np.zeros((2, self.total_gene_queries, self.Nd), dtype=np.float64)
        
        # Expand moments for broadcasting: (1, 1, Nd, S)
        exponents_broadcasted = exponent_grid[np.newaxis, np.newaxis, :, :]
        
        # Loop over chunks of gene queries for memory safety
        for q_start in tqdm.tqdm(range(0, self.total_gene_queries, query_chunk_size), disable=tqdm_disable):
            
            # select chunk of gene queries
            q_end = min(q_start + query_chunk_size, self.total_gene_queries)
            curr_q_len = q_end - q_start
            
            chunk_queries_A = queries_A_list[q_start:q_end]
            chunk_queries_B = queries_B_list[q_start:q_end]
            
            flat_A_indices = np.array(chunk_queries_A).flatten()
            flat_B_indices = np.array(chunk_queries_B).flatten()
            
            # Select and reshape count data for this chunk
            all_A_data = A_csr[:, flat_A_indices].toarray().reshape(N, curr_q_len, S_A)
            all_B_data = B_csr[:, flat_B_indices].toarray().reshape(N, curr_q_len, S_B)
            combined_base = np.concatenate([all_A_data, all_B_data], axis=2)
            
            # Moment matrix for query chunk
            local_Q_matrix = np.zeros((N, curr_q_len * self.Nd), dtype=np.float64)
            
            # Vectorised computation of all powers & products of count data needed for query chunk
            powers_tensor = np.power(combined_base[:, :, np.newaxis, :], exponents_broadcasted)
            query_all_moments = np.prod(powers_tensor, axis=3)
            local_Q_matrix[:, :] = query_all_moments.reshape(N, curr_q_len * self.Nd)
            
            # Store bootstrap moments for query chunk
            local_moments_flat = np.zeros((curr_q_len * self.Nd, resamples), dtype=np.float64)
            
            # Loop over chunks of bootstrap resamples for memory safety
            for b_start in range(0, resamples, bootstrap_chunk_size):

                # Select resampled indices for chunk
                b_end = min(b_start + bootstrap_chunk_size, resamples)
                curr_b_len = b_end - b_start
                chunk_inds = boot_indices[:, b_start:b_end]
                
                # Compute chunk bootstrap weights: proportion times each cell is resampled
                weights = np.zeros((N, curr_b_len), dtype=np.float64)
                for col in range(curr_b_len):
                    weights[:, col] = np.bincount(chunk_inds[:, col], minlength=N) / N
                
                # Product with Q pre-computed base powers to compute bootstrap moments
                local_moments_flat[:, b_start:b_end] = np.dot(local_Q_matrix.T, weights)
                
            # Reshape local bootstrap matrix to (curr_q_len, self.Nd, resamples)
            local_moments_3d = local_moments_flat.reshape(curr_q_len, self.Nd, resamples)
        
            # Compute confidence intervals for query chunk: can then discard large matrices of this query
            chunk_lb, chunk_ub = np.nanquantile(local_moments_3d, [(alpha / 2), 1 - (alpha / 2)], axis=2)
            
            # Store intervals
            moment_intervals[0, q_start:q_end, :] = chunk_lb
            moment_intervals[1, q_start:q_end, :] = chunk_ub

        # Store final confidence intervals
        self.moment_bounds = moment_intervals
