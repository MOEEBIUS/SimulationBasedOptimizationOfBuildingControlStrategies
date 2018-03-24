from propagate import propagate
import numpy as np

def pyOptCostConstraints(policy, **kwargs):
    dynModels = kwargs['args'][0] 
    T = kwargs['args'][1] 
    xlength = kwargs['args'][2] 
    states = kwargs['args'][3] 
    actions = kwargs['args'][4] 
    controlLim = kwargs['args'][5] 
    rand = kwargs['args'][6] 
    mc_samples = kwargs['args'][7] 
    explorationConstraint = kwargs['args'][8] 
    w = kwargs['args'][9] 
    comfort = kwargs['args'][10] 
    occ = kwargs['args'][11]
    costFunction = kwargs['args'][12]
    
    g = []
    
    
    xxProp,yyProp,ccProp,maxVar = propagate(dynModels, T, xlength, states, actions, policy, controlLim, rand, mc_samples, w, comfort, occ)
    
    if(costFunction == 1):
        f = np.linalg.norm(xxProp[:,actions])       # minEnergy
        f = 3 * f
    elif(costFunction == 2):                        # Tariff
        uu = xxProp[:,actions].copy()
        f = 0
#        print(uu)
        for ii in range(0,uu.shape[0]):
#            print(f)
            if(ii<6):
                f = f + 5*uu[ii,0]
            else:
                f = f + 15*uu[ii,0]
                
            
    g.append(maxVar - explorationConstraint)    # Exploration constraint
    g.append(np.sum(ccProp))                    # Comfort constraint

#    print("Cost Function = " + str(f))
#    print("Comfort Constraint = " + str(g[1]))
##    print("Variance = " + str(g[0]))
#    print(policy)
#    print("---------------------------------------------")
    
    fail = 0
    return f, g, fail

