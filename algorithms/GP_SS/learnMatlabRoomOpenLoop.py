import time
import numpy as np
import GPy
import pickle
from pylab import *
from pyOpt import *

from simulate import simulate
from pyOptCostConstraints import pyOptCostConstraints
import sys
# software paths
sys.path.append('../../../base')
sys.path.append('../../../utils')

from checkModelAccuracy import checkModelAccuracy
from generateRandomWeather import generateRandomWeather


""" Initialization """
t0 = time.time()
# states, actions
states = [0]
actions = [2]
disurbances = [1]
xlength = len(states)+len(actions)+len(disurbances)
policy = np.array(0.01*np.random.randn(len(states)+len(disurbances)+1),ndmin=2)
comfort = [20,24]
occ = [7,17]

# learning-related
initialRuns = 10
batchProcess = 10
internalLoops = 1

keepComfort = []
keepPower = []
keepWeather= []
keepBill = []
dt = 3600.0
T = 24
controlInputs = [0,]
controlLim = [0,2]
#controllerBounds = ((-2, 2), (-2, 2), (-2, 2))
controllerBounds = None
cobylaBounds = [controlLim, controlLim, controlLim, controlLim, 
                controlLim, controlLim, controlLim, controlLim, controlLim, 
                controlLim, controlLim, controlLim, controlLim, controlLim, 
                controlLim, controlLim, controlLim, controlLim
                ]
policy = np.array([0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2,
                   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],ndmin=2)
policyInit = np.array([0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2,
                   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],ndmin=2)
dynModels = []
mc_samples = 10000
num_inducing= 2000#int(initialRuns*T/3.0)#int(initialRuns / 5 * T)
explorationConstraint = 0.03

costFunction = 1 # 1: minEnergy; 2: Tariff; 3: TrackLoadProfile
if(costFunction == 1):
    baseResultsFolder = 'Results/Energy/'
elif(costFunction == 2):
    baseResultsFolder = 'Results/Cost/'

keepSameWeather = 1 # Use known pool of Weather files for comparison


    
""" Learning iterations """
for ii in range(0,batchProcess):
    if(keepSameWeather == 0):
        w = generateRandomWeather(T,dt)
    else:
        filename='Weathers/w'+str(ii+initialRuns)+'.out'
        # Lading the objects:
        with open(filename) as f:  # Python 3: open(..., 'rb')
            w = pickle.load(f)
        print('Using weather file: '+ 'Weathers/w'+str(ii+1)+'.out')
        
        
    """ Initial simulations to get data """
    for jj in range(0,initialRuns):
        rand = 1
        xx,yy,cc,uu,w,bill = simulate(T, states, actions, policy, controlLim, rand, w, dt, comfort, occ, costFunction)
        if(jj==0):
            x = xx
            y = yy
        else:
            x = np.concatenate((x,xx),axis=0)
            y = np.concatenate((y,yy),axis=0)
    
#    xt = np.hstack((x[:,0:2],np.ones((x.shape[0],1))))
#    a = np.dot(np.transpose(xt),xt)
#    b = np.linalg.pinv(a)
#    c = np.dot(b,np.transpose(xt))
#    d = np.dot(c,x[:,-1])
#    policy = np.array(d,ndmin=2)
#    print("Initial policy = " + str(policy))       

    
    """ Internal loops for optimization """
    for kk in range(0,internalLoops):
        policy = policyInit.copy()
        # learn dynamics model
        K1 = GPy.kern.Bias(input_dim = x.shape[1])
        K2 = GPy.kern.Linear(input_dim = x.shape[1],ARD=False)
        K3 = GPy.kern.RBF(x.shape[1],ARD=False)
        K4 = GPy.kern.Matern52(input_dim = x.shape[1],ARD=False)
        kernel = GPy.kern.Matern52(x.shape[1],ARD=False) #K1 + K2 + K3 + K4
        for jj in range(0,len(states)):
            if(x.shape[0] > num_inducing):
                print("Using Sparse GP Model...")
                dynModel = GPy.models.SparseGPRegression(x, y[:,jj].reshape(y.shape[0],1), kernel,
                                                   num_inducing = num_inducing)
                dynModels.append(dynModel.copy())
            else:
                print("Using Full GP Model...")
                dynModel = GPy.models.GPRegression(x, y[:,jj].reshape(y.shape[0],1), kernel)
                dynModel.optimize_restarts(num_restarts = 2)
                dynModel.optimize('bfgs', messages=True, max_iters=5000)
                dynModels.append(dynModel.copy())
            print(dynModels[jj])
            checkModelAccuracy(dynModel, x, y[:,jj], 0)
        
        
        # optimize
        rand = 0
        """ pyOpt optimization """
        # Instantiate Optimization Problem 
        opt_prob = Optimization('Matlab Room Constrained Problem',pyOptCostConstraints)
        opt_prob.addVar('x1','c',lower=cobylaBounds[0][0],upper=cobylaBounds[0][1],value=policy[0,0])
        opt_prob.addVar('x2','c',lower=cobylaBounds[1][0],upper=cobylaBounds[1][1],value=policy[0,1])
        opt_prob.addVar('x3','c',lower=cobylaBounds[2][0],upper=cobylaBounds[2][1],value=policy[0,2])
        opt_prob.addVar('x4','c',lower=cobylaBounds[3][0],upper=cobylaBounds[3][1],value=policy[0,3])
        opt_prob.addVar('x5','c',lower=cobylaBounds[4][0],upper=cobylaBounds[4][1],value=policy[0,4])
        opt_prob.addVar('x6','c',lower=cobylaBounds[5][0],upper=cobylaBounds[5][1],value=policy[0,5])
        opt_prob.addVar('x7','c',lower=cobylaBounds[6][0],upper=cobylaBounds[6][1],value=policy[0,6])
        opt_prob.addVar('x8','c',lower=cobylaBounds[7][0],upper=cobylaBounds[7][1],value=policy[0,7])
        opt_prob.addVar('x9','c',lower=cobylaBounds[8][0],upper=cobylaBounds[8][1],value=policy[0,8])
        opt_prob.addVar('x10','c',lower=cobylaBounds[9][0],upper=cobylaBounds[9][1],value=policy[0,9])
        opt_prob.addVar('x11','c',lower=cobylaBounds[10][0],upper=cobylaBounds[10][1],value=policy[0,10])
        opt_prob.addVar('x12','c',lower=cobylaBounds[11][0],upper=cobylaBounds[11][1],value=policy[0,11])
        opt_prob.addVar('x13','c',lower=cobylaBounds[12][0],upper=cobylaBounds[12][1],value=policy[0,12])
        opt_prob.addVar('x14','c',lower=cobylaBounds[13][0],upper=cobylaBounds[13][1],value=policy[0,13])
        opt_prob.addVar('x15','c',lower=cobylaBounds[14][0],upper=cobylaBounds[14][1],value=policy[0,14])
        opt_prob.addVar('x16','c',lower=cobylaBounds[15][0],upper=cobylaBounds[15][1],value=policy[0,15])
        opt_prob.addVar('x17','c',lower=cobylaBounds[16][0],upper=cobylaBounds[16][1],value=policy[0,16])
        opt_prob.addVar('x18','c',lower=cobylaBounds[17][0],upper=cobylaBounds[17][1],value=policy[0,17])
        opt_prob.addObj('f')
        opt_prob.addCon('g1','i')
        opt_prob.addCon('g2','i')
        print(opt_prob)
        
    
#        opt = NSGA2()
#        opt.setOption('PopSize', 360)
#        opt.setOption('maxGen', 500)
#        opt.setOption('PrintOut', 2)
#    #    opt.setOption('seed', 0.0)
#        
#        opt(opt_prob, 
#            args=[dynModels, T, xlength, states, actions, 
#                       controlLim, rand, mc_samples, explorationConstraint, 
#                       w, comfort, occ, costFunction])
#        print(opt_prob.solution(0))
#        a = opt_prob.solution(0)
#        for jj in range(0,policy.shape[1]):
#            policy[0,jj] = a.getVar(jj).value
            
#         # try
#        rand = 0
#        wbefore = w
#        xx,yy,cc,uu,w,bill = simulate(T, states, actions, policy, controlLim, rand, w, dt, comfort, occ, costFunction)
#        
#        print("GA Total Energy Consumption = " + str(np.sum(uu)))
#        print("GA Total Monetary Cost = " + str(np.sum(bill)))
#        print("GA Comfort Constraint = " + str(np.sum(cc)))
#        print("---------------------------------------------")
        
        
        opt = SLSQP()
        opt.setOption('ACC', 1.0e-20)
        opt.setOption('MAXIT', 1000)
        opt.setOption('IPRINT', 1)
    
        opt(opt_prob,# sens_step = 0.01, #.solution(0), store_hst=True,
                args=[dynModels, T, xlength, states, actions, 
                       controlLim, rand, mc_samples, explorationConstraint, 
                       w, comfort, occ, costFunction])
        

        print("Iteration: " + str(ii))   
#        a = opt_prob.solution(0).solution(0)
        a = opt_prob.solution(0)
        print(a)

        for jj in range(0,policy.shape[1]):
            policy[0,jj] = a.getVar(jj).value
        print(policy)
    
        
        # try
        rand = 0
        wbefore = w
        xx,yy,cc,uu,w,bill = simulate(T, states, actions, policy, controlLim, rand, w, dt, comfort, occ, costFunction)
       
        print("Internall Interation: " + str(kk)+ " || Total Energy Consumption = " + str(np.sum(uu)))
        print("Internall Interation: " + str(kk)+ " || Total Monetary Cost = " + str(np.sum(bill)))
        print("Internall Interation: " + str(kk)+ " || Comfort Constraint = " + str(np.sum(cc)))
        print("---------------------------------------------")
        
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(w, label='Weather')
    ax.plot(xx[:,0], label='Room Temperature')
    ax.plot(xx[:,2], label='Control Actions')
    ax.plot([1,T], [comfort[0], comfort[0]])
    ax.plot([1,T], [comfort[1], comfort[1]])
    ax.plot([occ[0],occ[0]], [comfort[0], comfort[1]])
    ax.plot([occ[1],occ[1]], [comfort[0], comfort[1]])
    ax.plot(uu, label='Energy Consumption')
    ax.plot(bill, label='Monetary Cost')
    plt.title('Constraints satisfaction')
    ax.legend()
    plt.draw()
    fig.savefig(baseResultsFolder+'GPConstraints'+str(ii+1)+'.svg', format='svg', dpi=1200)
    pickle.dump(fig, open(baseResultsFolder+'GPConstraints'+str(ii+1)+'.fig', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`
#    figx = pickle.load(open(baseResultsFolder+'GAConstraints'+str(ii+1)+'.fig', 'rb'))
#    figx.show() # Show the figure, edit it, etc.!
    plt.close(fig)
    
    x = np.concatenate((x,xx),axis=0)
    y = np.concatenate((y,yy),axis=0)
    keepComfort.append(np.sum(cc))
    keepPower.append(np.sum(uu))
    keepWeather.append(w)
    keepBill.append(np.sum(bill))

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(keepComfort, label='Comfort')
    ax.plot(keepBill, label='Monetary Cost')
    ax.plot(keepPower, label='Energy Consumption')
    plt.title('Convergence')
    ax.legend()
    plt.draw()
    fig.savefig(baseResultsFolder+'GPComfortEnergyCost'+str(ii+1)+'.svg', format='svg', dpi=1200)
    plt.close(fig)
    

    print("Total Energy Consumption = " + str(np.sum(uu)))
    print("Total Monetary Cost = " + str(np.sum(bill)))
    print("Comfort Constraint = " + str(np.sum(cc)))
    print("---------------------------------------------")
    
    
    """ Save all variables to a file """
    filename=baseResultsFolder+'GPworkspace'+str(ii+1)+'.out'
    # Saving the objects:
    with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([policy, keepComfort, keepPower, keepBill], f)


elapsed = (time.time() - t0)
print("Execution Time = " + str(elapsed) + " seconds...")































    
