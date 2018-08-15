########################################################################
# MUQ2 is required to run  https://bitbucket.org/mituq/muq2
########################################################################
"""
# Simulation functions 
#    Linear Regression
#    Banana Transformation
#    Parameters of Banana Transformation
"""
from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import numpy as np

# Import libEnsemble main
from libensemble.libE import libE
from libensemble.message_numbers import UNSET_TAG, STOP_TAG, PERSIS_STOP, EVAL_GEN_TAG, EVAL_SIM_TAG , FINISHED_PERSISTENT_GEN_TAG

# Import MUQ2 packages (ensure MUQ2 is in path)
import pymuqModeling as mm
import pymuqUtilities as mu
import pymuqApproximation as ma
import pymuqSamplingAlgorithms as ms # Needed for MCMC
from pymuqUtilities import RandomGenerator as rg

# For debugging, set a breakpoint with pdb.set_trace() or ipdb.set_trace()
#import pdb
#import ipdb

"""
Linear Regression
# MUQ class for forward evaluations 
# libE definition to evaluate the likelihood 
"""
def LinReg_libE(H, gen_info, sim_specs, libE_info):
    """
    Evaluates linear equation and the likelihood 
    """
    del libE_info # Ignored parameter
    batch = len(H['theta'])
    x = sim_specs['eval_at']
    data = sim_specs['data']
    O = np.zeros(batch, dtype=sim_specs['out'])
    fwdSolver = LinReg(x)
    noiseCov = sim_specs['noiseCov']
    likelihood = mm.Gaussian(data, noiseCov).AsDensity()
   
    for i,m in enumerate(H['theta']):
        sol = fwdSolver.Evaluate([m])[0]
        O['f'][i] = sol
        O['like'][i] = likelihood.Evaluate([sol])[0]

    return O, gen_info


class LinReg(mm.PyModPiece):
    def __init__(self, x):
        mm.PyModPiece.__init__(self, [2], # One input containing 2 components a,b
                                     [x.shape[0]]) # One output containing x.shape[0] components
        self.x  = x
        
    def EvaluateImpl(self, inputs):
        a, b = inputs[0]
        
        y = a*self.x + b
        
        self.outputs = [y] 



"""
Find Parameters of Banana Transformation
# Has two MUQ classes to evaluate the banaba transform and inverse banana transform 
# libE definition to evaluate the likelihood
"""        
def banana_param_libE(H, gen_info, sim_specs, libE_info):
    """
    Evaluates the inverse banana function and the likelihood 
    """
    batch = len(H['theta'])
    y = sim_specs['eval_at']
    n = sim_specs['n']
    data = sim_specs['data']
    O = np.zeros(batch, dtype=sim_specs['out'])
    invf = InvBananaTrans_param(y)
    noiseCov = sim_specs['noiseCov']
    likelihood = mm.Gaussian(data, noiseCov).AsDensity()
    xDist = mm.Gaussian(np.zeros((2)))
    
    for i,m in enumerate(H['theta']):
        log_place = np.zeros(y.shape[0])
        sol = invf.Evaluate([m])[0]
        sol = np.asarray([sol[:n],sol[n:]]).T
        O['x'][i] = sol
        for j in range(len(sol)):
            log_place[j] = np.exp(xDist.LogDensity(sol[j,:]))
        O['logVal'][i] = log_place
        sol_as_vect = np.append(np.append(sol[:,0], sol[:,1]), log_place)
        O['like'][i] = likelihood.Evaluate([sol_as_vect])[0]

    return O, gen_info

class BananaTrans_param(mm.PyModPiece):
    def __init__(self, z):
        mm.PyModPiece.__init__(self, [2], # One input containing 2 components
                                     [z.shape[0]*z.shape[1]]) # One output containing z.shape components
        self.z = z

    def EvaluateImpl(self, inputs):
        a, b = inputs[0]
        
        m = np.zeros(self.z.shape)
        m[:,0] = a * self.z[:,0]
        m[:,1] = self.z[:,1]/a - b*((a*self.z[:,0])**2 + a**2)
        
        self.outputs = [np.append(m[:,0],m[:,1])]
        
class InvBananaTrans_param(mm.PyModPiece):
    def __init__(self, m):
        mm.PyModPiece.__init__(self, [2], # One input containing 2 components
                                     [m.shape[0]*m.shape[1]]) # One output containing 2 components
        self.m = m
        
    def EvaluateImpl(self, inputs): #Impl stands for implement, remove Impl when call
        a, b = inputs[0]
        
        z = np.zeros(self.m.shape)
        z[:,0] = self.m[:,0]/a
        z[:,1] = self.m[:,1]*a + a*b*(self.m[:,0]**2 + a**2)
        self.outputs = [np.append(z[:,0],z[:,1])]
        
    def JacobianImpl(self, outDimWrt, inDimWrt, inputs):
        m = inputs[0]
        self.jacobian = np.array([ [1.0/a, 0], [2.0*a*b*self.m[:,0], a] ])
        
    def GradientImpl(self, outDimWrt, inDimWrt, inputs, sens):   
        m = inputs[0]
        self.gradient = np.dot(self.Jacobian(outDimWrt, inDimWrt, inputs),sens)
        



"""
Banana Transformation
# Has two MUQ classes to evaluate the banaba transform and inverse banana transform 
# libE definition to evaluate the log density
"""        
def banana_libE(H, gen_info, sim_specs, libE_info):
    """
    Evaluates the inverse banana function and the log density 
    """ 
    O = np.zeros(len(H['theta']), dtype=sim_specs['out'])
    invf = InvBananaTrans(sim_specs['a'],sim_specs['b'])
    logBanana = np.zeros(H['theta'].shape[0])
    xDist = mm.Gaussian(np.zeros(2))
    for i,m in enumerate(H['theta']):
        trans = invf.Evaluate([m])[0]
        O['logVal'][i] = xDist.LogDensity(trans)

    return O, gen_info

  
class InvBananaTrans(mm.PyModPiece):  
    def __init__(self, a, b):
        mm.PyModPiece.__init__(self, [2], # One input containing 2 components
                                     [2]) # One output containing 2 components
        self.a = a
        self.b = b
        
    def EvaluateImpl(self, inputs): #Impl stands for implement, remove Impl when call
        m = inputs[0]
        
        z = np.zeros((2))
        z[0] = m[0]/self.a
        z[1] = m[1]*self.a + self.a*self.b*(m[0]**2 + self.a**2)
        self.outputs = [z]
        
    def JacobianImpl(self, outDimWrt, inDimWrt, inputs):
        m = inputs[0]
        self.jacobian = np.array([ [1.0/self.a, 0], [2.0*self.a*self.b*m[0], self.a] ])
        
    def GradientImpl(self, outDimWrt, inDimWrt, inputs, sens):

        grad = np.zeros((2))
        grad[0] = v[0]/self.a + 2.0*self.a*self.b*m[0]*v[1]
        grad[1] = v[1]*self.a
        self.gradient = grad        
        
        
class BananaTrans(mm.PyModPiece): 
    def __init__(self, a, b):
        mm.PyModPiece.__init__(self, [2], # One input containing 2 components
                                     [2]) # One output containing 2 components
        
        self.a = a
        self.b = b
        
    def EvaluateImpl(self, inputs): #input and outout relationship of model
        z = inputs[0]
        
        m = np.zeros((2))
        m[0] = self.a * z[0]
        m[1] = z[1]/self.a - self.b*((self.a*z[0])**2 + self.a**2)
        
        self.outputs = [m] #list of vectors     
        
        
        