"""
# Generator functions 
#    + uniform_random_sample_persistent_gen
#        - Persisent uniform distribution generator
#        - Acts on output of sim_func
# 
# NOTE: This is an example of how libEnsemble works, not something to do in practice 
"""
from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import numpy as np

# Import libEnsemble main
from libensemble.libE import libE
from libensemble.message_numbers import UNSET_TAG, STOP_TAG, PERSIS_STOP, EVAL_GEN_TAG, EVAL_SIM_TAG , FINISHED_PERSISTENT_GEN_TAG

# For debugging, set a breakpoint with pdb.set_trace() or ipdb.set_trace()
#import pdb
#import ipdb


def uniform_random_sample_persistent_gen(H,persis_info,gen_specs,libE_info):
    """
    Generates points uniformly over the domain defined at first by gen_specs['ub'] and
    gen_specs['lb'] and subsequently by taking the minimum 'x' of each batch.
    """
    n = gen_specs['n']
    comm = libE_info['comm']
    while 1:
        # If you have received data from sim_func evaluation use it 
        if 'calc_in' in vars(): 
            ub = calc_in['x'] + 1.0*np.ones(n)
            lb = calc_in['x'] - 1.0*np.ones(n)
        else:
            ub = gen_specs['ub']
            lb = gen_specs['lb']
        
        b = gen_specs['gen_batch_size']
        
        # Receive information from the manager (or a STOP_TAG) 
        status = MPI.Status()
        O = np.zeros(b, dtype=gen_specs['out'])
        # Generate samples
        for i in range(0,b):
            x = persis_info['rand_stream'].uniform(lb,ub)
            O['x'][i] = x.flatten()
            D = {'calc_out':O, 
                 'libE_info': {'persistent':True},
                 'calc_status': UNSET_TAG,
                 'calc_type': EVAL_GEN_TAG
                }
        # Send generated samples   
        comm.send(obj=D,dest=0,tag=EVAL_GEN_TAG)
        # Check to see what information (if any) is coming
        comm.probe(source=0, tag=MPI.ANY_TAG, status=status) 
        tag = status.Get_tag()
        if tag in [STOP_TAG, PERSIS_STOP]:   
            break 
        else:
            Work = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        
        # Receive information
        libE_info = Work['libE_info']
        calc_in = comm.recv(buf=None, source=0)
    
    return O, persis_info, tag

