"""
# An example that approximates the value of Pi by generating 
# a large number of random points and see how many fall in the circle 
# enclosed by a square. Use that
#
#  Pi/4 = #_pts_in_circle/#_pts_total
#
# Demonstrates: 
#    - Basic generator and simulator function
#    - Manipulating and saving output of libEnsemble 
#
# Following command will run the code
# mpiexec -np 4 python3 calc_pi.py
# (-np # tell how many processors to use, it must be >=2 for this example)
# 
# Following command open up -np # of windows and allow for debuggind with pdb or ipdb
# mpiexec -np 4 xterm -e "python3 calc_pi.py.py"
"""

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys             # for adding to path
import numpy as np
import matplotlib.pyplot as plt

# Import libEnsemble main
from libensemble.libE import libE

# For debugging, set a breakpoint with pdb.set_trace() or ipdb.set_trace()
#import pdb
#import ipdb


######
# The gen_func and sim_func
# They can be separate files that you will need to import
######

# Gen_func
def uniform_random_sample(H,gen_info,gen_specs,libE_info):
    """
    Generates points uniformly over the domain defined by gen_specs['ub'] and
    gen_specs['lb'].
    """
    del libE_info # Ignored parameter

    ub = gen_specs['ub']
    lb = gen_specs['lb']
    b = gen_specs['gen_batch_size']

    O = np.zeros(b, dtype=gen_specs['out'])
    for i in range(0,b):
        x = gen_info['rand_stream'].uniform(lb,ub,(1,2))
        O['x'][i] = x

    return O, gen_info

# Sim_func
def in_circle(H, gen_info, sim_specs, libE_info):
    """
    Evaluates if the generated point is in the circle
    """
    batch = len(H['x'])
    O = np.zeros(batch,dtype=sim_specs['out'])

    for i,x in enumerate(H['x']):
        if (x[0]**2.0 + x[1]**2.0 <= sim_specs['r']**2.0):
            O['f'][i] = 1. # Pt. is in circle
        else:
            O['f'][i] = 0. # Pt. is not in circle

    return O, gen_info 

######
# The main code
######

r = 2. #radius of circle
# State the generating function, its arguments, output, and necessary parameters (and their sizes).
gen_specs = {'gen_f': uniform_random_sample, 
             'in': ['sim_id'],
             'out': [('x',float, 2),], # This will be given to the sim function
             'lb': -r, #lower bound
             'ub': r,  #upper bound
             'gen_batch_size': 10, # number of samples per batch
             }

# State the simulation (objective) function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': in_circle, # This is the function you want to evaluate
             'in': ['x'], # This is from the gen_func
             'out': [('f',float,1), ],                     
             'r' : r,
             }


# Tell libEnsemble when to stop
N = 5000
exit_criteria = {'sim_max': N} #Stop when N simulations have been done


persis_info = {}
for i in range(MPI.COMM_WORLD.Get_size()):
    persis_info[i] = {'rand_stream': np.random.RandomState(i)}
    
# Perform the run
H, gen_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info=persis_info)

if MPI.COMM_WORLD.Get_rank() == 0: # The manager is rank 0 by default and contains all of the information
    M = sum(H['f'])
    pi = 4.*M/N
    print ("The approximated value of Pi is {}".format(pi))
    #print(H)
    #print(H.dtype)
    #np.save('calc_pi', H) # Saves H as npy file
    ind = H['sim_id'][H['f'] == 1.] # Finds the indices where the generated point was in the circle
    x_pts = H['x'][ind,0]
    y_pts = H['x'][ind,1]
    plt.plot(x_pts,y_pts,'o')
    plt.show()
    





