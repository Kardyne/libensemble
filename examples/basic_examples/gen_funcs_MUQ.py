########################################################################
# MUQ2 is required to run  https://bitbucket.org/mituq/muq2
########################################################################
"""
# Generator functions that use MUQ2
"""
from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys, os             # for adding to path
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random


# Import libEnsemble main
from libensemble.libE import libE
from libensemble.message_numbers import UNSET_TAG, STOP_TAG, PERSIS_STOP, EVAL_GEN_TAG, EVAL_SIM_TAG , FINISHED_PERSISTENT_GEN_TAG


# Import MUQ
sys.path.insert(0,'/homes/kepetros/software/MUQ2_INSTALL/lib')

import pymuqModeling as mm
import pymuqUtilities as mu
import pymuqApproximation as ma
import pymuqSamplingAlgorithms as ms # Needed for MCMC
from pymuqUtilities import RandomGenerator as rg

# For debugging, set a breakpoint with pdb.set_trace() or ipdb.set_trace()
#import pdb
#import ipdb


def gaussian_Known_Target_adapt_prop_mean_DM_weighting(H,gen_info,gen_specs,libE_info):
    """
    Generator that draws samples from a gaussian proposal and evaluates the log density of points w.r.t the prior and proposal.
    When sim_func evaluations of samples are returned the weights are then calculated.
    Essentially doing Importance Sampling
    Inputs:
        H: Not used
        gen_info: Needed for libE
        gen_specs:
            subbatch_size - How many samples to do in a sub batch, int
            num_subbatches - Number of sub batches, int 
            mu_init - Mean of proposals, should be same number as num_subbatches, list of arrays
            sigma_init - Proposal covariance matrix, array
            bounds - Bounds of uniform prior, array
        libE_info: Needed for libE 
    Output:
        theta - Samples
        prior - Uniform prior log density of samples
        prop - Gaussian proposal log density of samples
        subbatch - Which subbatch the samples were generated in 
        batch - Which batch the samples were generated in 
        weight - The wieghts of the samples 
    """
    logPropMu = gen_specs['mu_init']
    dim = len(logPropMu[0])
    logPropCov = gen_specs['sigma_init']
    
    comm = libE_info['comm']

    # Receive information from the manager (or a STOP_TAG) 
    status = MPI.Status()
    batch = -1
    while 1:
        batch += 1            
        O = np.zeros(gen_specs['subbatch_size']*gen_specs['num_subbatches'], dtype=gen_specs['out'])
        # Send weights with next batch of samples in order to see them in the output
        if 'w' in vars(): 
            O['weight'] = w
        row = -1

        prop = []
        for j in range(gen_specs['num_subbatches']):
            logProp = mm.Gaussian(logPropMu[j], logPropCov).AsDensity()
            prop.append(logProp)
            for i in range(0,gen_specs['subbatch_size']):
                row += 1
                theta = logProp.Sample()
                O['theta'][row] = theta
                O['subbatch'][row] = j
                O['batch'][row] = batch
                
        for i in range(len(O['theta'])):
            prior_sum = 0
            for j in range(gen_specs['num_subbatches']):
                prior_sum = prior_sum + prop[j].Evaluate([O['theta'][i]])[0]
            O['prop'][i] = prior_sum
        
        # What is being sent to manager to pass on to workers
        D = {'calc_out':O,
             'libE_info': {'persistent':True},
             'calc_status': UNSET_TAG,
             'calc_type': EVAL_GEN_TAG
            }
        
        # Sending Data to manager
        comm.send(obj=D,dest=0,tag=EVAL_GEN_TAG)

       # Probing manager to see what type of data is being sent
        comm.probe(source=0, tag=MPI.ANY_TAG, status=status)          
        tag = status.Get_tag()
        if tag in [STOP_TAG, PERSIS_STOP]:   
            break 
        else: 
            Work = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        
        # Recieving data from manager
        libE_info = Work['libE_info']
        calc_in = comm.recv(buf=None, source=0)
        # Calculating the weights
        w = np.exp(calc_in['logVal'] + np.log(gen_specs['num_subbatches']) - O['prop']) 
        logPropMu = []
        for p in range(gen_specs['num_subbatches']):
            norm_w = w[calc_in['subbatch']==p]/np.nansum(w[calc_in['subbatch']==p])
            w[calc_in['subbatch']==p] = norm_w
            inds = residual_resample(norm_w)
            ind = inds[np.random.randint(len(inds))]# pick an index from inds uniformly need subbatch size+1
            logPropMu.append(O['theta'][ind].flatten())
    return O, gen_info, tag


def residual_resample(weights):
    """ Performs the residual resampling algorithm used by particle filters.

    Based on observation that we don't need to use random numbers to select
    most of the weights. Take int(N*w^i) samples of each particle i, and then
    resample any remaining using a standard resampling algorithm [1]


    Parameters
    ----------

    weights : list-like of float (must be normalized)
        list of weights as floats

    Returns
    -------

    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.

    References
    ----------

    .. [1] J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic
       systems. Journal of the American Statistical Association,
       93(443):1032–1044, 1998.
    """

    N = len(weights)
    indexes = np.zeros(N, 'i')
    if np.sum(weights) != 1.:
        weights = weights/np.sum(weights)   

    # take int(N*w) copies of each weight, which ensures particles with the
    # same weight are drawn uniformly
    num_copies = (np.floor(N*np.asarray(weights))).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]): # make n copies
            indexes[k] = i
            k += 1

    # use multinormal resample on the residual to fill up the rest. This
    # maximizes the variance of the samples
    residual = weights - num_copies     # get fractional part
    residual /= sum(residual)           # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1. # avoid round-off errors: ensures sum is exactly one
    indexes[k:N] = np.searchsorted(cumulative_sum, random(N-k))

    return indexes



















