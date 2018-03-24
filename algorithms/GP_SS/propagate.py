import numpy as np
import GPy
from numpy import linalg as LA
from controller import controller
from comfortConstraints import comfortConstraints

import sys
# software paths
sys.path.append('../../../base')
sys.path.append('../../../utils')
from sample_posterior_custom import sample_posterior_custom

def propagate(dynModels, T, xlength, states, actions, policy, controlLim, rand, mc_samples, w1, comfort, occ):
    w = w1.copy()
#    model = dynModel.copy()
    xx = np.zeros((T+1,len(states)+w.shape[1]))
    uu = np.zeros((T+1,len(actions)))
    cc = np.zeros((T+1,1))
    var = np.zeros((T,len(states)))
    xx[0,] = [17, w[0,]]
    uu[0,] = controller(policy,xx[0,],controlLim,rand,0)
    cc[0,] = comfortConstraints(xx[0,states], comfort, 0, occ)
    
    # v1 with averaging per time-step
    for ii in range(1,T+1):      
        newX = np.hstack([xx[ii-1,],uu[ii-1]]).reshape(1,xx.shape[1]+uu.shape[1])
        for jj in range(0,len(states)):
            results = dynModels[jj].predict(newX)
            xx[ii, jj] = results[0]
            var[ii-1, jj] = results[1]
#        for jj in range(0,len(states)):
#            c = dynModels[jj].posterior_samples_f(newX, mc_samples)
#            xx[ii, jj] = np.sum(c) / mc_samples
#            var[ii-1, jj] = np.sum(LA.norm(c-xx[ii,jj])) / (mc_samples-1)
        xx[ii,1] = w[ii,]
        uu[ii,] = controller(policy,xx[ii,],controlLim,rand,ii)
        cc[ii,] = comfortConstraints(xx[ii,states], comfort, ii, occ)
        
        
    x = np.concatenate((xx,uu),axis=1)
   
    xx = x[0:-1,]
    yy = x[1:,states] - xx[:,states]
    maxVar = np.mean(var)

    return xx,yy,cc,maxVar

