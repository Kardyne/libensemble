"""
# An example that samples the rosenbrock function  
# Demonstrates: 
#    - The use of an allocation function
#    - A persistent generator
#    - Using data from the sim_func inform the gen_func
# 
# Following command will run the code
# mpiexec -np 4 python3 persistent_generator_acts_on_feedback.py
# (-np # tell how many processors to use, it must be >=3 for this example)
#
# Following command open up -np # of windows and allow for debuggind with pdb or ipdb
#   mpiexec -np 4 xterm -e "python3 persistent_generator_acts_on_feedback.py"
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


# For debugging, set a breakpoint with pdb.set_trace() or ipdb.set_trace()
import pdb
#import ipdb

######
# The gen_func, sim_func, and alloc_func
# They can be separate files that you will need to import
######

# Gen_func
def uniform_random_sample(H,persis_info,gen_specs,libE_info):
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

# Sim_func
def rosenbrock(H, gen_info, sim_specs, libE_info):
    """
    Evaluates Rosenbrock Function
    """
    batch = len(H['x'])
    O = np.zeros(batch,dtype=sim_specs['out'])

    for i,x in enumerate(H['x']):
        O['f'][i] = sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


    return O, gen_info 

# Alloc_func
def persistent_gen(W, H, sim_specs, gen_specs, persis_info):
    """ 
    Decide what should be given to workers. Note that everything put into
    the Work dictionary will be given, so we are careful not to put more gen or
    sim items into Work than necessary.


    This allocation function will 
    - Start up a persistent generator.
    - It will only do this if at least one worker will be left to perform
      simulation evaluations.
    - If multiple starting points are available, the one with smallest function
      value is chosen. 
    - If no candidate starting points exist, points from existing runs will be
      evaluated (oldest first)
    - If no points are left, call the gen_func 
    """

    # Initialize Work as empty
    Work = {}
    gen_count = sum(W['persis_state'] == EVAL_GEN_TAG)
    already_in_Work = np.zeros(len(H),dtype=bool) # To mark points as they are included in Work, but not yet marked as 'given' in H.


    ###### Persistent Generator ######
    # If i is idle, but in persistent mode, and its calculated values have
    # returned, give them back to i. Otherwise, give nothing to i
    for i in W['worker_id'][np.logical_and(W['active']==0,W['persis_state']!=0)]:
        gen_inds = H['gen_worker']==i 
        if np.all(H['returned'][gen_inds]): # Has sim_f completed everything from this persistent worker?
            # Then give back the 'x' that gave the minimum value
            last_batch_inds = H['f'][gen_inds]==np.min(H['f'][gen_inds])
            inds_to_send_back = np.where(np.logical_and(gen_inds,last_batch_inds))[0] 
            Work[i] = {'persis_info': persis_info[i],
                       'H_fields': sim_specs['in'],
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(inds_to_send_back),
                                     'persistent': True
                                }
                       }
    # Workers
    for i in W['worker_id'][np.logical_and(W['active']==0,W['persis_state']==0)]:
        """perform sim evaluations from existing runs (if they exist)."""
        q_inds_logical = np.logical_and.reduce((~H['given'],~H['paused'],~already_in_Work))

        if np.any(q_inds_logical):
            sim_ids_to_send = np.nonzero(q_inds_logical)[0][0] # oldest point

            Work[i] = {'H_fields': sim_specs['in'],
                       'persis_info': {}, # Our sims don't need information about how points were generated
                       'tag':EVAL_SIM_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(sim_ids_to_send),
                                },
                      }

            already_in_Work[sim_ids_to_send] = True
        else:
            """ Finally, generate points since there is nothing else to do. """
            if gen_count > 0: 
                continue
            gen_count += 1
            """ There are no points available, so we call our gen_func (this only gets called once when 
            persistent generator is assigned  """
            Work[i] = {'persis_info': persis_info[i],
                       'H_fields': gen_specs['in'],
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': [],
                                     'persistent': True
                                }

                       }

    return Work, persis_info


n = 2 # Dimension of input
# State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': rosenbrock, # This is the function whose output is being minimized
             'in': ['x'], # keys will be given to the sim_func
             'out': [('f',float),],# This is the output from the sim_func
             }


# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': uniform_random_sample,
             'in': [], 
             'out': [('x',float,n),],
             'lb': -5*np.ones(n), #lower bound
             'ub': 5.*np.ones(n),  #upper bound
             'gen_batch_size': 10,
             'n': n,
             }


alloc_specs = {'out':[], 'alloc_f': persistent_gen}

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 5000}

persis_info = {}
for i in range(MPI.COMM_WORLD.Get_size()):
    persis_info[i] = {'rand_stream': np.random.RandomState(i)}

# Perform the run
H, gen_info, flag = libE(sim_specs, gen_specs, exit_criteria, alloc_specs=alloc_specs, persis_info=persis_info)


if MPI.COMM_WORLD.Get_rank() == 0: # The manager is rank 0 by default and contains all of the information
    # 2D
    #np.save('rose', H)
    plt.figure()
    h= plt.scatter(H['x'][:,0], H['x'][:,1], c=H['f'], alpha=0.5)
    plt.colorbar(h)
    plt.show()
    # 3D
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #h= ax.scatter(H['x'][:,0],H['x'][:,1], H['x'][:,2], c=H['f'], alpha=0.3)
    #plt.colorbar(h)
    #plt.show()
    
