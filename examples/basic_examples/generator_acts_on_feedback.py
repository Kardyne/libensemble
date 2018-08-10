"""
# An example that samples the rosenbrock function  
# Demonstrates: 
#    - The use of an allocation function
#    - A persistent generator
#    - Using data from the sim_func inform the gen_func
# 
# Following command will run the code
# mpiexec -np 4 python3 generator_acts_on_feedback.py
# (-np # tell how many processors to use, it must be >=3 for this example)
#
# Following command open up -np # of windows and allow for debuggind with pdb or ipdb
#   mpiexec -np 4 xterm -e "python3 generator_acts_on_feedback.py"
"""

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys             # for adding to path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import libEnsemble main
from libensemble.libE import libE
from libensemble.message_numbers import UNSET_TAG, STOP_TAG, PERSIS_STOP, EVAL_GEN_TAG, EVAL_SIM_TAG , FINISHED_PERSISTENT_GEN_TAG
from libensemble.sim_funcs.six_hump_camel import six_hump_camel
# For debugging, set a breakpoint with pdb.set_trace() or ipdb.set_trace()
import pdb
#import ipdb

######
# The gen_func, sim_func, and alloc_func
# They can be separate files that you will need to import
######

# Gen_func
def normal_random_sample(H,persis_info,gen_specs,libE_info):
    """
    Generates points uniformly over the domain defined at first by gen_specs['ub'] and
    gen_specs['lb'] and subsequently by taking the minimum 'x' of each batch.
    """
    b = gen_specs['gen_batch_size']
    var = gen_specs['var']
    if len(H['f']) != 0:
       ind = np.argmin(H['f'])
       mean = H['x'][ind]
       var = 0.9*var
    else:
        mean = gen_specs['mean']
        
        
    O = np.zeros(b, dtype=gen_specs['out'])
    for i in range(0,b):
        x = persis_info['rand_stream'].normal(mean,var)
        O['x'][i] = x

    return O, persis_info


def three_hump_camel(H, persis_info, sim_specs, _):
    """
    Evaluates the three hump camel function
    """

    batch = len(H['x'])
    O = np.zeros(batch,dtype=sim_specs['out'])

    for i,x in enumerate(H['x']):
        x1 = x[0]
        x2 = x[1]
        term1 = 2*x1**2
        term2 = -1.05*x1**4
        term3 = x1**6/6.
        term4 = x1*x2
        term5 = x2**2
        O['f'][i] = term1 + term2 + term3 + term4 +term5

    return O, persis_info


# State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': three_hump_camel, # This is the function whose output is being minimized
             'in': ['x'], # keys will be given to the sim_func
             'out': [('f',float),],# This is the output from the sim_func
             }


# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': normal_random_sample,
             'in': ['sim_id'] + ['f'] + ['x'], 
             'out': [('x',float,2),],
             'mean': np.array([-5,5]), #lower bound
             'var': 20.,  #upper bound
             'gen_batch_size': 5,
             'batch_mode': True,
             'update': 0,
             }


# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 6000}

persis_info = {}
for i in range(MPI.COMM_WORLD.Get_size()):
    persis_info[i] = {'rand_stream': np.random.RandomState(i)}

# Perform the run
H, gen_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info=persis_info)


if MPI.COMM_WORLD.Get_rank() == 0: # The manager is rank 0 by default and contains all of the information
    # 2D
    #np.save('rose', H)
    plt.figure()
    h= plt.scatter(H['x'][:,0], H['x'][:,1], c=np.log(H['f']), alpha=0.5,s=2)
    plt.colorbar(h)
    plt.show()
    # 3D
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #h= ax.scatter(H['x'][:,0],H['x'][:,1], H['x'][:,2], c=H['f'], alpha=0.3)
    #plt.colorbar(h)
    #plt.show()
    
