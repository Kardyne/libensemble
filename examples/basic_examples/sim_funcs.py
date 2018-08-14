"""
# Simulation functions 
#     Rosenbrock Function
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


def rosenbrock(H, gen_info, sim_specs, libE_info):
    """
    Evaluates Rosenbrock Function
    """
    batch = len(H['x'])
    O = np.zeros(batch,dtype=sim_specs['out'])

    for i,x in enumerate(H['x']):
        O['f'][i] = sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


    return O, gen_info 


