'''
Module to handle correlation computations.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import scipy
import tqdm
import numpy as np

# ------------------------------------------------
# Correlation function
# ------------------------------------------------

def compute_correlations(dataset, method="OB", confidence=0.95, resamples=1000, query_chunk_size=100, bootstrap_chunk_size=100, tqdm_disable=False):
    '''
    Compute point estimate and bootstrap percentile confidence intervals on
    OB & AL correlation coefficients for each gene query in dataset

    Arguments:
        dataset: dataset object containing count information
        confidence: percentile confidence level
        resamples: number of bootstrap resamples
        query_chunk_size: division of gene queries for memory safety
        bootstrap_chunk_size: division of bootstrap resamples for memory safety
        tqdm_disable: bool to hide progress bar

    Returns
        correlations_OB: (_, 3) array of OB point & interval estimates
        correlations_OG: (_, 3) array of OG point & interval estimates
    '''

    # helpful values
    sparse_A = dataset.sparse_A
    sparse_B = dataset.sparse_B
    queries_A_list = [query[0] for query in dataset.gene_queries]
    queries_B_list = [query[1] for query in dataset.gene_queries]
    beta = dataset.beta
    S_A = len(queries_A_list[0])
    S_B = len(queries_B_list[0])
    S = S_A + S_B
    alpha = 1 - confidence

    # only implemented for pairwise correlation
    if not (S_A == 1 and S_B == 1):
        raise Exception("Correlation between more than 2 species not currently supported")
    
    # size
    N = dataset.cells

    # initialize random generator
    rng = np.random.default_rng()
    
    # Flatten input query lists to ensure they are 1D arrays of scalar indices
    flat_A_indices = np.array(queries_A_list).flatten()
    flat_B_indices = np.array(queries_B_list).flatten()
    
    # Capture efficiency moments
    E_beta = np.mean(beta)
    E_beta2 = np.mean(beta**2)
    
    # Convert sparse input arrays to CSR for high-speed batch column slicing
    A_csr = scipy.sparse.csr_matrix(sparse_A)
    B_csr = scipy.sparse.csr_matrix(sparse_B)
    
    # Bootstrap resamples indices
    boot_indices = rng.integers(0, N, size=(N, resamples))
    
    # Store final correlations: (point, CI lb, CI ub)
    correlation_OB_results = np.zeros((dataset.total_gene_queries, 3), dtype=np.float64)
    correlation_OG_results = np.zeros((dataset.total_gene_queries, 3), dtype=np.float64)
    
    # Loop over chunks of gene queries for memory safety
    for q_start in tqdm.tqdm(range(0, dataset.total_gene_queries, query_chunk_size), disable=tqdm_disable):

        # select chunk of gene queries
        q_end = min(q_start + query_chunk_size, dataset.total_gene_queries)
        curr_q_len = q_end - q_start
        
        # Select count data for this chunk
        X_block = A_csr[:, flat_A_indices[q_start:q_end]].toarray() # Shape: (N, curr_q_len)
        Y_block = B_csr[:, flat_B_indices[q_start:q_end]].toarray() # Shape: (N, curr_q_len)
        
        # Moment matrix of 5 moments needed per query: (N, curr_q_len * 5)
        local_Q = np.zeros((N, curr_q_len * 5), dtype=np.float64)
        local_Q[:, 0::5] = X_block          # X
        local_Q[:, 1::5] = Y_block          # Y
        local_Q[:, 2::5] = X_block**2       # X^2
        local_Q[:, 3::5] = Y_block**2       # Y^2
        local_Q[:, 4::5] = X_block * Y_block # X * Y
        
        # Point estimates: average over cell axis
        orig_OB = np.mean(local_Q, axis=0).reshape(curr_q_len, 5)
        
        E_x_OB, E_y_OB = orig_OB[:, 0], orig_OB[:, 1]
        E_x2_OB, E_y2_OB, E_xy_OB = orig_OB[:, 2], orig_OB[:, 3], orig_OB[:, 4]

        varx_OB = E_x2_OB - E_x_OB**2
        vary_OB = E_y2_OB - E_y_OB**2

        # Compute correlation OB point estimate
        valid_mask = (varx_OB > 0.0) & (vary_OB > 0.0)
        point_corrs = np.full(curr_q_len, np.nan)
        point_corrs[valid_mask] = (E_xy_OB[valid_mask] - E_x_OB[valid_mask]*E_y_OB[valid_mask]) / (
            np.sqrt(varx_OB[valid_mask]) * np.sqrt(vary_OB[valid_mask])
        )
        correlation_OB_results[q_start:q_end, 0] = point_corrs
        
        # Convert OB to OG moments
        E_xy_OG = E_xy_OB / E_beta2
        E_x_OG  = E_x_OB / E_beta
        E_y_OG  = E_y_OB / E_beta
        varx_OG = ((1 / E_beta2)*E_x2_OB + (1 / E_beta)*E_x_OB - (1 / E_beta2)*E_x_OB) - E_x_OG**2
        vary_OG = ((1 / E_beta2)*E_y2_OB + (1 / E_beta)*E_y_OB - (1 / E_beta2)*E_y_OB) - E_y_OG**2

        # Compute correlation OG point estimate
        valid_mask = (varx_OG > 0.0) & (vary_OG > 0.0)
        point_corrs = np.full(curr_q_len, np.nan)
        point_corrs[valid_mask] = (E_xy_OG[valid_mask] - E_x_OG[valid_mask]*E_y_OG[valid_mask]) / (
            np.sqrt(varx_OG[valid_mask]) * np.sqrt(vary_OG[valid_mask])
        )
        correlation_OG_results[q_start:q_end, 0] = point_corrs
        
        # Bootstrap moments for query chunk: (curr_q_len * 5, resamples)
        local_boot_flat = np.zeros((curr_q_len * 5, resamples), dtype=np.float64)
        
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
            local_boot_flat[:, b_start:b_end] = np.dot(local_Q.T, weights)
            
        # Reshape local bootstrap matrix to (curr_q_len, 5, resamples)
        boot_moments = local_boot_flat.reshape(curr_q_len, 5, resamples)
        
        E_x_OB_b, E_y_OB_b = boot_moments[:, 0, :], boot_moments[:, 1, :]
        E_x2_OB_b, E_y2_OB_b, E_xy_OB_b = boot_moments[:, 2, :], boot_moments[:, 3, :], boot_moments[:, 4, :]

        varx_OB_b = E_x2_OB_b - E_x_OB_b**2
        vary_OB_b = E_y2_OB_b - E_y_OB_b**2

        # Vectorized calculation of all bootstrap correlation estimates
        boot_estimates = np.full((curr_q_len, resamples), np.nan)
        boot_mask = (varx_OB_b > 0.0) & (vary_OB_b > 0.0)
        
        boot_estimates[boot_mask] = (E_xy_OB_b[boot_mask] - E_x_OB_b[boot_mask]*E_y_OB_b[boot_mask]) / (
            np.sqrt(varx_OB_b[boot_mask]) * np.sqrt(vary_OB_b[boot_mask])
        )
        
        # Compute confidence intervals for query chunk: can then discard large matrices of this query
        chunk_lb, chunk_ub = np.nanquantile(boot_estimates, [(alpha / 2), 1 - (alpha / 2)], axis=1)
        
        # Store intervals
        correlation_OB_results[q_start:q_end, 1] = chunk_lb
        correlation_OB_results[q_start:q_end, 2] = chunk_ub
        
        # Convert OB to OG moments
        E_xy_OG_b = E_xy_OB_b / E_beta2
        E_x_OG_b  = E_x_OB_b / E_beta
        E_y_OG_b  = E_y_OB_b / E_beta
        varx_OG_b = ((1 / E_beta2)*E_x2_OB_b + (1 / E_beta)*E_x_OB_b - (1 / E_beta2)*E_x_OB_b) - E_x_OG_b**2
        vary_OG_b = ((1 / E_beta2)*E_y2_OB_b + (1 / E_beta)*E_y_OB_b - (1 / E_beta2)*E_y_OB_b) - E_y_OG_b**2
        
        # Vectorized calculation of all bootstrap correlation estimates
        boot_estimates = np.full((curr_q_len, resamples), np.nan)
        boot_mask = (varx_OG_b > 0.0) & (vary_OG_b > 0.0)
        
        boot_estimates[boot_mask] = (E_xy_OG_b[boot_mask] - E_x_OG_b[boot_mask]*E_y_OG_b[boot_mask]) / (
            np.sqrt(varx_OG_b[boot_mask]) * np.sqrt(vary_OG_b[boot_mask])
        )
        
        # Compute confidence intervals for query chunk: can then discard large matrices of this query
        chunk_lb, chunk_ub = np.nanquantile(boot_estimates, [(alpha / 2), 1 - (alpha / 2)], axis=1)
        
        # Store intervals
        correlation_OG_results[q_start:q_end, 1] = chunk_lb
        correlation_OG_results[q_start:q_end, 2] = chunk_ub
        
    return correlation_OB_results, correlation_OG_results
