########################################################################
# MUQ2 is required to run  https://bitbucket.org/mituq/muq2
# Pandas is optional for plotting
########################################################################
# """
# Runs MUQ2 and libEnsemble in order to generate the banana distribution
#
#    m = [a*z1
#         1/a*z2+b(a*z1)^2 + b*a^2 ]
#
# using importance sampling, adaptive Metropolis Hastings MCMC, and adaptive importance sampling
#
# Execute via the following command:
#    mpiexec -np 4 python3 banana.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# 
# Following command opens up -np # of windows and allow for debuggind with pdb or ipdb
#   mpiexec -np 4 xterm -e "python3 banana.py"
# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys, os             # for adding to path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

# For debugging, set a breakpoint with pdb.set_trace() or ipdb.set_trace()
#import pdb
#import ipdb

# To save dictionary
#import pickle

# Import libEnsemble main
from libensemble.libE import libE
from libensemble.message_numbers import UNSET_TAG, STOP_TAG, PERSIS_STOP, EVAL_GEN_TAG, EVAL_SIM_TAG , FINISHED_PERSISTENT_GEN_TAG

# Import MUQ2 packages (ensure MUQ2 is in path)
import pymuqModeling as mm
import pymuqUtilities as mu
import pymuqApproximation as ma
import pymuqSamplingAlgorithms as ms # Needed for MCMC
from pymuqUtilities import RandomGenerator as rg

from alloc_funcs import persistent_gen_logval as alloc_func
from gen_funcs_MUQ import gaussian_Known_Target_adapt_prop_mean_DM_weighting as gen_func
from sim_funcs_MUQ import BananaTrans, InvBananaTrans, banana_libE 


#################
### Main Code ###
#################

paramDim = 2
plot = True
zDist = mm.Gaussian(np.zeros((paramDim)))
n = 50000
a = 1.0
b = 1.0
if MPI.COMM_WORLD.Get_rank() == 0:
    invf = InvBananaTrans(a,b)
    fwd = BananaTrans(a,b)
    x = np.zeros((n,2))
    y = np.zeros((n,2))
    sampLogVals = np.zeros(n)
    for i in range(n):
        x[i,:] = zDist.Sample()
        y[i,:] = fwd.Evaluate([x[i,:]])[0]
        sampLogVals[i] = np.exp(zDist.LogDensity(x[i,:]))

# Number of samples
numSamps = 50000
propMu = np.array([0,-4])
propCov = np.array([ [4, 0],
                     [0, 20]])

compare = False
if MPI.COMM_WORLD.Get_rank() == 0 and compare==True:
    graph = mm.WorkGraph()
    graph.AddNode(invf, "Banana Transformation")
    graph.AddNode(zDist.AsDensity(), "Gaussian Reference")
    graph.AddEdge("Banana Transformation", 0, "Gaussian Reference", 0)
    bananaDens = graph.CreateModPiece("Gaussian Reference")

#####################################
# Importance Sampling
#####################################
if MPI.COMM_WORLD.Get_rank() == 0 and compare==True:
    isProp = mm.Gaussian(propMu, propCov)
    w = np.zeros((numSamps))
    samp = np.zeros((paramDim,numSamps))
    for i in range(numSamps):
        samp[:,i] = isProp.Sample()
        logBanana = bananaDens.Evaluate( [samp[:,i]])[0] 
        logProp = isProp.LogDensity(samp[:,i])
        w[i] = np.exp(logBanana - logProp)

    thres = 1e-4
    inds = np.where(w>thres)[0]
    num_small = w.shape[0]-inds.shape[0]
    H_IS = {'weight': w,
            'sample': samp,
            'thres': thres,
            'small_w': num_small,
            'num_samps': numSamps,
           }

    #outfile = open('banana_IS','wb')
    #pickle.dump(H_IS,outfile)
    #outfile.close()

    plt.figure()
    h = plt.scatter(samp[0,:],samp[1,:], c=w, alpha = 0.5, s=1.)
    plt.colorbar(h)
    plt.title("IS Banana Dist")

    plt.figure()
    h = plt.scatter(samp[0,inds],samp[1,inds], c=w[inds], alpha = 0.5, s=1.)
    plt.colorbar(h)
    plt.title("IS Banana Dist no small")

    plt.show()


#####################################
# Adaptive Metropolis MCMC
#####################################
if MPI.COMM_WORLD.Get_rank() == 0 and compare==True:
    problem = ms.SamplingProblem(bananaDens)

    proposalOptions = dict()
    proposalOptions['Method'] = 'AMProposal'
    proposalOptions['ProposalCovariance'] = propCov
    proposalOptions['AdaptSteps'] = 1000
    proposalOptions['AdaptStart'] = 1000
    proposalOptions['AdaptScale'] = 0.1

    kernelOptions = dict()
    kernelOptions['Method'] = 'MHKernel'
    kernelOptions['Proposal'] = 'ProposalBlock'
    kernelOptions['ProposalBlock'] = proposalOptions

    options = dict()
    options['NumSamples'] = numSamps
    options['ThinIncrement'] = 1
    options['BurnIn'] = 0
    options['KernelList'] = 'Kernel1'
    options['PrintLevel'] = 3
    options['Kernel1'] = kernelOptions

    mcmc = ms.SingleChainMCMC(options,problem)

    startPt = propMu
    samps = mcmc.Run(startPt)

    sampMat = samps.AsMatrix()
    plt.figure()
    plt.plot(sampMat[0,:], label=('x'))
    plt.plot(sampMat[1,:],label=('y'))
    plt.legend()
    plt.title("AM Banana Dist Chains")

    plt.figure()
    plt.scatter(sampMat[0,:], sampMat[1,:], c=samps.Weights(), s=1.)
    plt.title("AM Banana Dist")


    fig = plt.figure(figsize=(12,6))

    for i in range(paramDim):
        shiftedSamp = sampMat[i,:]-np.mean(sampMat[i,:])
        corr = np.correlate(shiftedSamp, shiftedSamp, mode='full')
        plt.plot(corr[int(corr.size/2):]/np.max(corr), label='Dimension %d'%i)
    
    maxLagPlot = 1500
    plt.axhline(y=0, linewidth=.5, color='k')
    plt.xlim([0,maxLagPlot])
    plt.ylim([-0.1,1.1])
    plt.legend()
    plt.title("AM Banana Dist Autocorrelation")



    df = pd.DataFrame(sampMat.T, columns=['x', 'y'])
    scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')

    df = pd.DataFrame(y, columns=['x', 'y'])
    scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')

    plt.show()

    H_AM = {'weight': samps.Weights(),
            'sample': sampMat,
            'num_samps': numSamps,
            'ESS': samps.ESS()
           }

 
    #outfile = open('banana_AM','wb')
    #pickle.dump(H_AM,outfile)
    #outfile.close()


# State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': banana_libE, 
             'in': ['theta'], 
             'a': a,
             'b': b,
             'out': [('logVal',float, 1)
                    ],
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': gen_func,
             'in': [],
             'out': [('theta',float,2),('batch',int),('subbatch',int),('prop',float,1), ('weight',float,1)],
             'mu_init': [propMu, np.array([15, 0.])], #mean  
             'sigma_init': propCov, #covariance
             'subbatch_size': 10,
             'num_subbatches': 2,
             'adapt_step': 1000,
             }


alloc_specs = {'out':[], 
               'alloc_f': alloc_func}

# How many batches to run
num_batches = 5000

# Tell libEnsemble when to stop 
exit_criteria = {'sim_max': gen_specs['num_subbatches']*num_batches} #To get correct exit criteria 

np.random.seed(1)
persis_info = {}
for i in range(MPI.COMM_WORLD.Get_size()):
    persis_info[i] = {'rand_stream': np.random.RandomState(i)}


# Can't do a "persistent gen run" if only one worker
if MPI.COMM_WORLD.Get_size()<=2:
    quit() 

# Perform the run
H, gen_info, flag = libE(sim_specs, gen_specs, exit_criteria, alloc_specs=alloc_specs, persis_info=persis_info)

if MPI.COMM_WORLD.Get_rank() == 0:
    # Change the last weights to correct values (H is a list on other cores and only array on manager)
    ind = 2*gen_specs['subbatch_size']*gen_specs['num_subbatches']
    H['weight'][-ind:] = np.exp(H['logVal'][-ind:] - H['prop'][-ind:])
    norm_weights = H['weight']/np.nansum(H['weight'])
    if plot ==True:
        plt.figure()
        h = plt.scatter(y[:,0],y[:,1], c=sampLogVals, alpha = 0.5, s=1.)
        plt.colorbar(h)
        plt.title("True Banana Dist")
        inds = np.where(np.log(H['weight'])>-10)[0]
        
        plt.figure()
        h = plt.scatter(H['theta'][:,0],H['theta'][:,1], c=H['weight'], alpha=0.5, s=1.)
        plt.colorbar(h)
        plt.title("AIS Banana Dist")
        
        plt.figure()
        h = plt.scatter(H['theta'][inds,0],H['theta'][inds,1], c=H['weight'][inds], alpha=0.5, s=1.)
        plt.colorbar(h)
        plt.title("AIS no small weights Banana Dist")
        
        plt.figure()
        h = plt.scatter(H['theta'][:,0],H['theta'][:,1], c=norm_weights, alpha=0.5, s=1.)
        plt.colorbar(h)
        plt.title("AIS Banana Dist norm weights")
        
        plt.figure()
        h = plt.scatter(H['theta'][inds,0],H['theta'][inds,1], c=norm_weights[inds], alpha=0.5, s=1.)
        plt.colorbar(h)
        plt.title("AIS no small and norm weights Banana Dist")
        
        plt.figure()
        h = plt.scatter(H['theta'][:,0],H['theta'][:,1], c=np.log(H['weight']), alpha=0.5, s=1.)
        plt.colorbar(h)
        plt.title("AIS Banana Dist log")
        
        plt.figure()
        h = plt.scatter(H['theta'][inds,0],H['theta'][inds,1], c=np.log(H['weight'][inds]), alpha=0.5, s=1.)
        plt.colorbar(h)
        plt.title("AIS no small weights Banana Dist log")
        
        fig = plt.figure(figsize=(12,6))
        for i in range(2):
            shiftedSamp = H['theta'][:,i]-np.mean(H['theta'][:,i])
            corr = np.correlate(shiftedSamp, shiftedSamp, mode='full')
            plt.plot(corr[int(corr.size/2):]/np.max(corr), label='Dimension %d'%i)
    
        maxLagPlot = 1500
        plt.plot([-maxLagPlot,0.0],[4.0*maxLagPlot,0.0],'--k', label='Zero')

        plt.xlim([0,maxLagPlot])
        plt.ylim([-0.1,1.1])
        plt.legend()
        
        
        df = pd.DataFrame(H['theta'][inds,:], columns=['x', 'y'])
        scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
        
        plt.show()
    #np.save('banana', H)    
