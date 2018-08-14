"""
# Allocation functions 
#   + persistent_gen_min
#       - assign a persistent generator
#       - send the 'x' that gave the minimum        
#   + persistent_gen_like
#       - assign a persistent generator
#       - send all the likelihoods and subbatch numbers  
#   + persistent_gen_banana
#       - assign a persistent generator
#       - send all the logvalues and subbatch numbers    
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


def persistent_gen_min(W, H, sim_specs, gen_specs, persis_info):
    """ 
    Decide what should be given to workers. Note that everything put into
    the Work dictionary will be given, so we are careful not to put more gen or
    sim items into Work than necessary.

    This allocation function will 
    - Start up a persistent generator.
    - It will only do this if at least one worker will be left to perform
      simulation evaluations.
    - Return the 'x' from the batch that has the minimum evaluated value
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
                       'H_fields': sim_specs['in'], #specify keys of what you want to send back
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(inds_to_send_back), #specify what rows you want to send back
                                     'persistent': True
                                }
                       }
    
    ###### Workers ######
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


def persistent_gen_like(W, H, sim_specs, gen_specs, persis_info):
    """ 
    These persistent generators produce points (x) in batches and subbatches. 
    The points x are given in subbatches to workers to perform a calculation.
    When all subbatches have returned, their output is given back to the
    corresponding persistent generator.
    
    The first time called there are no persistent workers so the 1st for loop is not done 
    """

    Work = {}
    gen_count = sum(W['persis_state'] == EVAL_GEN_TAG)
    already_in_Work = np.zeros(len(H),dtype=bool) # To mark points as they are included in Work, but not yet marked as 'given' in H.
    
    # If i is idle, but in persistent mode, and generated work has all returned
    # give output back to i. Otherwise, give nothing to i
    for i in W['worker_id'][np.logical_and(W['active']==0,W['persis_state']!=0)]:
        gen_inds = H['gen_worker']==i
        if np.all(H['returned'][gen_inds]): # Has sim_f completed everything from this persistent worker?
            # Then give back everything in the last batch
            last_batch_inds = H['batch'][gen_inds]==np.max(H['batch'][gen_inds])
            inds_to_send_back = np.where(np.logical_and(gen_inds,last_batch_inds))[0] 
            # Assign weights to correct batch (last two batchs weights will need to be done outside of alloc_func)
            if H['batch'][-1] > 0:
                n = gen_specs['subbatch_size']*gen_specs['num_subbatches']
                k = H['batch'][-1]
                H['weight'][(n*(k-1)):(n*k)] = H['weight'][(n*k):(n*(k+1))]    
            Work[i] = {'persis_info': persis_info[i],
                       'H_fields': ['like'] + ['subbatch'], #specify keys of what you want to send back
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(inds_to_send_back), #atleast_1d -> Convert inputs to arrays with at least one dimension.
                                     'persistent': True
                                }
                       }

    
    for i in W['worker_id'][np.logical_and(W['active']==0,W['persis_state']==0)]:    
        # perform sim evaluations (if any point hasn't been given).
        q_inds_logical = np.logical_and(~H['given'],~already_in_Work) 
        if np.any(q_inds_logical):
            sim_ids_to_send = np.nonzero(q_inds_logical)[0][H['subbatch'][q_inds_logical]==np.min(H['subbatch'][q_inds_logical])]
            Work[i] = {'H_fields': sim_specs['in'], #things to evaluate
                       'persis_info': {}, # Our sims don't need information about how points were generated
                       'tag':EVAL_SIM_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(sim_ids_to_send), #tells me what x's the returned values go with
                                },
                      }

            already_in_Work[sim_ids_to_send] = True

        else:
            # Finally, generate points since there is nothing else to do. 
            if gen_count > 0: 
                continue # continue with the next loop of the iteration
            gen_count += 1
            # There are no points available, so we call our gen_func
            Work[i] = {'persis_info':persis_info[i],
                       'H_fields': gen_specs['in'],
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': [],
                                     'persistent': True
                                }
                       }

    return Work, persis_info


def persistent_gen_logval(W, H, sim_specs, gen_specs, persis_info):
    """ 
    Starts up to gen_count number of persistent generators.
    These persistent generators produce points (x) in batches and subbatches. 
    The points x are given in subbatches to workers to perform a calculation.
    When all subbatches have returned, their output is given back to the
    corresponding persistent generator.
    
    The first time called there are no persistent workers so the 1st for loop is not done 
    """

    Work = {}
    gen_count = sum(W['persis_state'] == EVAL_GEN_TAG)
    already_in_Work = np.zeros(len(H),dtype=bool) # To mark points as they are included in Work, but not yet marked as 'given' in H.
    
    # If i is idle, but in persistent mode, and generated work has all returned
    # give output back to i. Otherwise, give nothing to i
    for i in W['worker_id'][np.logical_and(W['active']==0,W['persis_state']!=0)]: 
        gen_inds = H['gen_worker']==i #it there is more than 1 persistant generator make sure you assign the correct work to it 
        if np.all(H['returned'][gen_inds]): # Has sim_f completed everything from this persistent worker?
            # Then give back everything in the last batch
            last_batch_inds = H['batch'][gen_inds]==np.max(H['batch'][gen_inds])
            inds_to_send_back = np.where(np.logical_and(gen_inds,last_batch_inds))[0] 
            # Assign weights to correct batch (last two batchs weights will need to be done outside of alloc_func)
            if H['batch'][-1] > 0:
                n = gen_specs['subbatch_size']*gen_specs['num_subbatches']
                k = H['batch'][-1]
                H['weight'][(n*(k-1)):(n*k)] = H['weight'][(n*k):(n*(k+1))]    
            Work[i] = {'persis_info': persis_info[i],
                       'H_fields': ['logVal'] + ['subbatch'], #specify keys of what you want to send back
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(inds_to_send_back), #atleast_1d -> Convert inputs to arrays with at least one dimension.
                                     'persistent': True
                                }
                       }

    for i in W['worker_id'][np.logical_and(W['active']==0,W['persis_state']==0)]:    
        # perform sim evaluations (if any point hasn't been given).
        q_inds_logical = np.logical_and(~H['given'],~already_in_Work) 
        if np.any(q_inds_logical):
            sim_ids_to_send = np.nonzero(q_inds_logical)[0][H['subbatch'][q_inds_logical]==np.min(H['subbatch'][q_inds_logical])]
            Work[i] = {'H_fields': sim_specs['in'], #things to evaluate
                       'persis_info': {}, # Our sims don't need information about how points were generated
                       'tag':EVAL_SIM_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(sim_ids_to_send), #tells me what x's the returned values go with
                                    },
                      }

            already_in_Work[sim_ids_to_send] = True

        else:
            # Finally, generate points since there is nothing else to do. 
            if gen_count  > 0: 
                continue # continue with the next loop of the iteration
            gen_count += 1
            # There are no points available, so we call our gen_func
            Work[i] = {'persis_info':persis_info[i],
                       'H_fields': gen_specs['in'],
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': [],
                                     'persistent': True
                                }

                       }

    return Work, persis_info