'''
Module to simulate synthetic data for testing inference methods.

Simulate stochastic reaction network models using the gillespie algorithm to
produce stationary samples emulating single-cell RNA sequencing observations.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

# ------------------------------------------------
# Gillespie simulation functions
# ------------------------------------------------

def coeff(xs, vsr):
    br = math.factorial(xs) / math.factorial(xs - vsr)
    return br

def gillespie(stoch_inp, stoch_out, rates, initial, tmax):
    '''Gillespie simulation algorithm'''

    # size
    R, S = stoch_inp.shape

    # reaction changes
    stoch = stoch_out - stoch_inp

    # initial
    if initial is None:
        initial = np.zeros(S)

    # initialization
    t = 0
    x = initial
    path = [x]
    path_times = [0]

    # convert to int
    x = x.astype(np.int64)
    stoch = stoch.astype(np.int64)
    stoch_inp = stoch_inp.astype(np.int64)

    # simulate for burn in and intervals between samples
    while t < tmax:

        # compute reaction propensities ----------------------------------------
        props = np.zeros(R)

        # for each reaction
        for r in range(R):

            # inital rate
            ar = rates[r]

            # for each species
            for s in range(S):
                
                # not involved
                if stoch_inp[r, s] == 0:
                    continue

                # not enough
                if x[s] < stoch_inp[r, s]:
                    ar = 0
                    continue

                # otherwise: multiply rate component
                ar *= coeff(x[s], stoch_inp[r, s])

            # set propensity
            props[r] = ar

        # overall propensity
        a0 = np.sum(props)

        # holding time ---------------------------------------------------------
        t_hold = -np.log(rng.uniform()) / a0
        t += t_hold
        path_times.append(t)

        # next reaction --------------------------------------------------------
        r = rng.choice(range(R), p=props / a0)
        x += stoch[r, :]
        path.append(list(x))

    # convert to array
    path = np.array(path)
    path_times = np.array(path_times)

    return path, path_times

def uniform_time_samples(path, path_times, tmin, tmax, tint, n):

    # size
    _, S = path.shape

    # store
    sampled_path = np.zeros((n, S))

    # for each time find state of step function
    t_sample = tmin
    for i in range(n):
        mask = (path_times < t_sample)
        sampled_path[i, :] = path[mask, :][-1, :]
        t_sample += tint

    return sampled_path

# ------------------------------------------------
# Simulation
# ------------------------------------------------

def simulate(stoch_inp, stoch_out, rates, initial=None, tmin=10, tmax=None, tint=None, n=None):
    '''
    Simulation of a SRN

    Arguments:
        stoch_inp: (R, S) array of stoichiometric input coefficients per reaction and species
        stoch_out: (R, S) array of stoichiometric output coefficients per reaction and species
        rates: (R) array of reaction rate constants per reaction
        initial: (S) array of initial molecule counts per species
        tmin: burn in time until path sampled
        
        Require at least 2 of the following:
            tmax: maximum time to simulate for
            tint: time interval between samples
            n: number of samples to take (tint time intervals)

    Returns
        sample: (n, S) array of n (approximately) stationary samples of the SRN
    '''

    # compute values
    if (tmax is not None):
        if (tint is not None):
            n = (tmax - tmin) // tint
        elif (n is not None):
            tint = (tmax - tmin) / n
    elif (tint is not None):
        if (n is not None):
            tmax = tmin + (n * tint)

    # gillespie path simulation
    path, path_times = gillespie(stoch_inp, stoch_out, rates, initial, tmax)

    # sample path a tint intervals
    sample = uniform_time_samples(path, path_times, tmin, tmax, tint, n)

    return sample
