#==============================================================================
# This project has received funding from the European Union's Horizon 2020 
# research and innovation programme under grant agreement No 680517 (MOEEBIUS)
# 
# Copyright (c) 2018 Technische Hochschule Nuernberg Georg Simon Ohm
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#==============================================================================

from __future__ import absolute_import, division, print_function
from copy import deepcopy

from Building import Building
import numpy as np
from pyOpt import SLSQP, NSGA2, Optimization
import GPy

class Algorithm():
    """ The high-level Algorithm Class. It defines an abstract optimization setup.
    The main API methods are:
        configure
        optimize
    """

    def configure(self, building):
        """Configure the optimization setup.
        Args:
            building (Building Class): a specific building instantiation
        """
        raise NotImplementedError

    def optimize(self, options = []):
        """Optimizes the control parameters to improve the performance of the
        building.
        Args: 
            options (dictionary): algorithm-specific parameters (e.g. max iterations)
            
        Returns: 
            policy (numpy array): the best controller found
            fscore (double): the best value for the cost function
            cscore (double): the amount of constraint violation
        """
        raise NotImplementedError
        
    def plotResults(self):
        """Plotting function."""
        
    def saveResults(self, baseResultsFolder):
        """Function for saving the optimization results.
        Args:
            baseResultsFolder (string): a folder for saving the results
        """
   
     
class SLSQP_pyOpt(Algorithm):
    """ Utilization of the SLSQP algorithm of pyOpt package."""
     
    def __init__(self):
        """Initialize the SLSQP algorithm for a specific building.
        """
        print("Initializing the SLSQP Optimizer...")
        self.opt_prob = []
        self.opt = []
        self.building = Building()
      
        
    def configure(self, building):
        
        # Get building and optimization setup properties
        self.building = deepcopy(building)
        self.T, self.states, self.actions, self.disturbances, self.controlLim, self.actionLim, self.comfort, self.occ, self.nvars, self.ncons = self.building.getConfiguration()
         
        # Define Box Constraints (min/max values) for the control parameters
        boxConstraints = []
        for ii in range(self.nvars):
            boxConstraints.append(self.controlLim)
             
        # Link to the python function calculating the cost and the constraints
        self.opt_prob = Optimization('SLSQP Constrained Problem', self.wrapSimulation)
        
        # Setupt Box Constrains in pyOpt
        for ii in range(self.nvars):
            self.opt_prob.addVar('x'+str(ii+1), 'c' , lower = boxConstraints[ii][0], upper = boxConstraints[ii][1], value = self.building.policy[0, ii])
         
        # Setupt Cost Function in pyOpt
        self.opt_prob.addObj('f')
         
        # Setupt Inequality Constraints in pyOpt
        for ii in range(self.ncons):
            self.opt_prob.addCon('g'+str(ii+1),'i')
             
        # Print the Optimization setup
        print("----------------------------------------")
        print("----------------------------------------")
        print("SLSQP Optimization setup:")
        print(self.opt_prob)
        
        
    def optimize(self, options = []):
        # Set SLSQP as the optimizer
        self.opt = SLSQP()
        
        # Set optimization options
        if(len(options) > 0):
            for ii in range(len(options)):
                self.opt.setOption(options.keys()[ii], options.values()[ii])
                
        # Print the Optimizer Options
        print("----------------------------------------")
        print("----------------------------------------")
        print("SLSQP Optimizer options:")
        print(self.opt.options)
        
        # Get optimized controller
        self.opt(self.opt_prob, sens_step = 1e-6)
        print(self.opt_prob.solution(0))
        a =  self.opt_prob.solution(0)
        for ii in range(self.building.policy.shape[1]):
            self.building.policy[0, ii] = a.getVar(ii).value
        
        return self.building.policy
    
    def wrapSimulation(self, policy):
        """A function that runs a building simulation and wraps the results in the 
        format required by PyOpt library.
        Args: 
            policy (numpy array): the controller to be used for the simulation
            
        Returns: 
            f (float): the cost function value
            g (list): the vales of all constraints
            fail (0/1): indicates if the function finished successfully
        """
        # Cost and Constraints
        f = 0
        g = []
        fail = 0
        
        # Run building simulation
        x, cost, constraints = self.building.simulate(policy)
        f = np.sum(cost)
        g.append(np.sum(constraints))
        
#        print(f)
#        print(g[0])
        
        return f, g, fail
         

class NSGA2_pyOpt(Algorithm):
    """ Utilization of the NSGA2 algorithm of pyOpt package."""
     
    def __init__(self):
        """Initialize the NSGA2 algorithm for a specific building.
        """
        print("Initializing the NSGA2 Optimizer...")
        self.opt_prob = []
        self.opt = []
        self.building = Building()
      
        
    def configure(self, building):
        
        # Get building and optimization setup properties
        self.building = deepcopy(building)
        self.T, self.states, self.actions, self.disturbances, self.controlLim, self.actionLim, self.comfort, self.occ, self.nvars, self.ncons = self.building.getConfiguration()
         
        # Define Box Constraints (min/max values) for the control parameters
        boxConstraints = []
        for ii in range(self.nvars):
            boxConstraints.append(self.controlLim)
             
        # Link to the python function calculating the cost and the constraints
        self.opt_prob = Optimization('NSGA2 Constrained Problem', self.wrapSimulation)
        
        # Setupt Box Constrains in pyOpt
        for ii in range(self.nvars):
            self.opt_prob.addVar('x'+str(ii+1), 'c' , lower = boxConstraints[ii][0], upper = boxConstraints[ii][1], value = self.building.policy[0, ii])
         
        # Setupt Cost Function in pyOpt
        self.opt_prob.addObj('f')
         
        # Setupt Inequality Constraints in pyOpt
        for ii in range(self.ncons):
            self.opt_prob.addCon('g'+str(ii+1),'i')
             
        # Print the Optimization setup
        print("----------------------------------------")
        print("----------------------------------------")
        print("NSGA2 Optimization setup:")
        print(self.opt_prob)
        
        
    def optimize(self, options = []):
        # Set NSGA2 as the optimizer
        self.opt = NSGA2()
        
        # Set optimization options
        if(len(options) > 0):
            for ii in range(len(options)):
                self.opt.setOption(options.keys()[ii], options.values()[ii])
                
        # Print the Optimizer Options
        print("----------------------------------------")
        print("----------------------------------------")
        print("NSGA2 Optimizer options:")
        print(self.opt.options)
        
        # Get optimized controller
        self.opt(self.opt_prob)
        print(self.opt_prob.solution(0))
        a =  self.opt_prob.solution(0)
        for ii in range(self.building.policy.shape[1]):
            self.building.policy[0, ii] = a.getVar(ii).value
        
        return self.building.policy
    
    def wrapSimulation(self, policy):
        """A function that runs a building simulation and wraps the results in the 
        format required by PyOpt library.
        Args: 
            policy (numpy array): the controller to be used for the simulation
            
        Returns: 
            f (float): the cost function value
            g (list): the vales of all constraints
            fail (0/1): indicates if the function finished successfully
        """
        # Cost and Constraints
        f = 0
        g = []
        fail = 0
        
        # Run building simulation
        x, cost, constraints = self.building.simulate(policy)
        f = np.sum(cost)
        g.append(np.sum(constraints))
        
#        print(f)
#        print(g[0])
        
        return f, g, fail
        
    
class GP_SS(Algorithm):
    """ Implementation of Building Identification and Control using Gaussian Process
    State-Space models. A GP regression model per state is created and then the 
    constrained optimization problem is solved using PyOpt's SLSQP solver."""
    
    def __init__(self):
        """Initialize the GP_SS algorithm for a specific building.
        """
        print("Initializing the GP_SS...")
        self.building = Building()
        self.X = []
        self.Y = []
        self.dynModels = []
        # Number of Initial Exploration Simulation
        self.initExploration = 36
        # Number of Samples for witching to Sparse Gaussian Processes
        self.num_inducing = 2000
        # Safety constraint for exploration
        self.explorationConstraint = 0.03
        
        # Needed to determine the best controller out of all iterations
        self.costs = []
        self.constraints = []
        self.policies = []
        self.baseline = []
    
        
    def configure(self, building):
        # Get building and optimization setup properties
        self.building = deepcopy(building)
        self.T, self.states, self.actions, self.disturbances, self.controlLim, self.actionLim, self.comfort, self.occ, self.nvars, self.ncons = self.building.getConfiguration()
        
        # Run initial random simulations and contstruct initial dataset
        for ii in range(0, self.initExploration):
            self.building.rand = 1
            xa, cf, cc = self.building.simulate(self.building.policy)
            xx = xa[0 : -1,]
            yy = xa[1:, self.states]
            if(ii == 0):
                self.X = xx
                self.Y = yy
            else:
                self.X = np.concatenate((self.X, xx), axis=0)
                self.Y = np.concatenate((self.Y, yy), axis=0)
                
        # Last simulation is the baseline controller
        self.building.rand = 0
        xa, cf, cc = self.building.simulate(self.building.policy)
        self.baseline.append(self.building.policy)
        self.baseline.append(np.sum(cf))
        self.baseline.append(np.sum(cc))
        xx = xa[0 : -1,]
        yy = xa[1:, self.states]
        self.X = np.concatenate((self.X, xx), axis=0)
        self.Y = np.concatenate((self.Y, yy), axis=0)
        
        
        
    def optimize(self, options):
        # Set max number of optimization iterations
        maxIter = 1
        if(len(options) > 0):
            maxIter = options["MAXIT"]
            
        # Log costs and constraints for all internal iterations
        self.costs = np.zeros((maxIter + 1, 1))
        self.constraints = np.zeros((maxIter + 1, self.ncons))
        self.policies = np.zeros((maxIter + 1, self.nvars))
        
        # GP_SS process
        initPolicy = self.building.policy.copy()
        for ii in range(maxIter):
            # Train GP state-space models
            kernel = GPy.kern.Matern52(self.X.shape[1], ARD=False)
            for jj in range(0, len(self.states)):
                if(self.X.shape[0] > self.num_inducing):
                    print("Using Sparse GP Model...")
                    dynModel = GPy.models.SparseGPRegression(self.X, self.Y[:,jj].reshape(self.Y.shape[0], 1), kernel, num_inducing = self.num_inducing)
                    self.dynModels.append(dynModel.copy())
                else:
                    print("Using Full GP Model...")
                    dynModel = GPy.models.GPRegression(self.X, self.Y[:,jj].reshape(self.Y.shape[0], 1), kernel)
                    dynModel.optimize_restarts(num_restarts = 2)
                    dynModel.optimize('bfgs', messages=True, max_iters=5000)
                    self.dynModels.append(dynModel.copy())
                print(self.dynModels[jj])
                self.checkModelAccuracy(dynModel, self.X, self.Y[:,jj])
                
            # Define Box Constraints (min/max values) for the control parameters
            boxConstraints = []
            for jj in range(self.nvars):
                boxConstraints.append(self.controlLim)
            
            # Link to the python function calculating the cost and the constraints. Note that 
            # this is not the actual simulation, but the propagate function
            self.opt_prob = Optimization('GPSS_SLSQP Constrained Problem', self.propagate)
            
            # Setupt Box Constrains in pyOpt
            for jj in range(self.nvars):
                self.opt_prob.addVar('x' + str(jj + 1), 'c' , lower = boxConstraints[jj][0], upper = boxConstraints[jj][1], value = self.building.policy[0, jj])
             
            # Setupt Cost Function in pyOpt
            self.opt_prob.addObj('f')
             
            # Setupt Inequality Constraints in pyOpt
            for jj in range(self.ncons + 1):
                self.opt_prob.addCon('g'+str(jj + 1),'i')
                 
            # Print the Optimization setup
            print("----------------------------------------")
            print("----------------------------------------")
            print("GPSS_SLSQP Optimization setup:")
            print(self.opt_prob)
            
            optionsSLSQP = {'ACC': 1.0e-20, 'MAXIT': 10000, 'IPRINT': 1}
            # Set SLSQP as the optimizer
            self.opt = SLSQP()
            
            # Set optimization options
            for jj in range(len(optionsSLSQP)):
                self.opt.setOption(optionsSLSQP.keys()[jj], optionsSLSQP.values()[jj])
                    
            # Print the Optimizer Options
            print("----------------------------------------")
            print("----------------------------------------")
            print("SLSQP Optimizer options:")
            print(self.opt.options)
            
            # Get optimized controller
            self.opt(self.opt_prob, sens_step = 1e-6)
            print(self.opt_prob.solution(0))
            a =  self.opt_prob.solution(0)
            for jj in range(self.building.policy.shape[1]):
                self.building.policy[0, jj] = a.getVar(jj).value
                

            # Evaluate the optimized controller in the simulation model
            xa, cf, cc = self.building.simulate(self.building.policy)
            print("COST: = =========== " + str(np.sum(cf)))
            print("CONSTRAINT: = =========== " + str(np.sum(cc)))
            xx = xa[0 : -1,]
            yy = xa[1:, self.states]
            if(ii == 0):
                self.X = xx
                self.Y = yy
            else:
                self.X = np.concatenate((self.X, xx), axis=0)
                self.Y = np.concatenate((self.Y, yy), axis=0)
                
            self.costs[ii, 0] = np.sum(cf)
            for jj in range(self.ncons):
                self.constraints[ii, jj] = np.sum(cc[:, jj])
            
            self.policies[ii, :] = self.building.policy.copy()
                
            self.building.policy = initPolicy.copy()
            
        self.policies[ii + 1, :] = self.baseline[0].copy()
        self.costs[ii + 1, 0] = self.baseline[1]
        self.constraints[ii + 1, 0] = self.baseline[2]
        
        policyIndex = self.selectBestController(self.costs, self.constraints)
        self.building.policy = self.policies[policyIndex, :].copy()
                
        return self.building.policy
        
    
    def selectBestController(self, costs, constraints):
        """A function that selects the best controller out of a set of controllers, 
        based on their performance on the cost function and the constraints.
        Args: 
            costs (numpy array): The resulting cost of each controller, as evaluated
            on the simulation model
            constraints (numpy array): The resulting constraints of each controller, as evaluated
            on the simulation model
            
        Returns: 
            policyIndex (float): the index of the best controller
        """
        wCost = 1
        wConstraints = 10000000
        p = np.zeros((costs.shape[0], 1))
        for ii in range(costs.shape[0]):
            p[ii, 0] = p[ii, 0] + wCost * costs[ii, 0]
            for jj in range(constraints.shape[1]):
                p[ii, 0] = p[ii, 0] + wConstraints * constraints[ii, jj]
        policyIndex = np.argmin(p)
        return policyIndex
        
    
    def propagate(self, policy):
        """A function that uses the GP state-space models identified from the data 
        to perform rollouts based on a given controller.
        Args: 
            policy (numpy array): the controller to be used for the rollout
            
        Returns: 
            f (float): the cost function value
            g (list): the vales of all constraints
            fail (0/1): indicates if the function finished successfully
        """
        
        # Initial state
        xx = np.zeros((self.building.T + 1, len(self.building.states) + self.building.w.shape[1]))
        uu = np.zeros((self.building.T + 1, len(self.building.actions)))
        cc = np.zeros((self.building.T + 1, 1))
        cf = np.zeros((self.building.T + 1, 1))
        var = np.zeros((self.building.T, len(self.building.states)))
        xx[0,] = self.X[0, 0: 2] # [17, w[0,]]
        uu[0,] = self.building.controller(policy, xx[0,], 0)
        cc[0,] = self.building.comfortConstraints(xx[0, self.building.states], 0)
        cf[0,] = self.building.costFunction(uu[0,], 0)
        
        # state propagation using the provided controller
        for ii in range(1, self.building.T+1):      
            newX = np.hstack([xx[ii - 1,], uu[ii - 1]]).reshape(1, xx.shape[1] + uu.shape[1])
            for jj in range(0, len(self.building.states)):
                results = self.dynModels[jj].predict(newX)
                xx[ii, jj] = results[0]
                var[ii-1, jj] = results[1]
            xx[ii,1] = self.building.w[ii,]
            uu[ii,] = self.building.controller(policy, xx[ii,], ii)
            cc[ii,] = self.building.comfortConstraints(xx[ii, self.building.states], ii)
            cf[ii,] = self.building.costFunction(uu[ii,], ii)
        
        f = np.sum(cf)
        g = []    
        g.append(np.mean(var) - self.explorationConstraint)    # Exploration constraint
        g.append(np.sum(cc))
        fail = 0
        
        return f, g, fail
    
    
    def checkModelAccuracy(self, dynModel, xtest, ytest):
        """A function that evaluates the accuracy of the GP state-space model.
        Args: 
            dynModel (GPy object): The GP model
            xtest (numpy array): The features of the regression 
            ytest (numpy array): The targets of the regression
        """
        
        results = dynModel.predict(xtest)
        ypred = results[0]
#        sGP = results[1]
         
        rsqTrain, maeTrain, rsqAdjTrain = self.evaluateGoodnessOfFit(xtest, ytest, ypred)
        print("Rsq train Gaussian Processes Regression = " + str(rsqTrain))
        print("Rsq Adjusted train Gaussian Processes Regression = " + str(rsqAdjTrain))
        print("MAE train Gaussian Processes Regression = " + str(maeTrain))
        
        
    def evaluateGoodnessOfFit(self, x, y, ypred):
        """A function that evaluates the goodness of fit, under different measures.
        Args: 
            x (numpy array): The features of the regression 
            y (numpy array): The targets of the regression
            ypred (numpy array): The predictions of the regression model
        
        Returns: 
            Rsquared (float): R-squared
            mae (float): Mean Absolute Error
            rsqAdj (float): The Adjusted R-square
        """
        
        print(y.shape[0])
        print(x.shape[1])
        y_hat = np.mean(y)
        SStot = np.sum(np.power(y - y_hat,2))
        SSres = np.sum(np.power(y - ypred.flatten(),2))
        if(SStot == 0):
            rsq = 1
        else:
            rsq = 1 - SSres/SStot
        mae = np.sum(np.abs(y - ypred.flatten()))/ypred.shape[0]
        
        rsqAdj = 1 - (1-rsq) * (y.shape[0]-1) / (y.shape[0]-x.shape[1]-1)
        
        return rsq, mae, rsqAdj
        
    
    def wrapSimulation(self, policy):
        """A function that runs a building simulation and wraps the results in the 
        format required by PyOpt library.
        Args: 
            policy (numpy array): the controller to be used for the simulation
            
        Returns: 
            f (float): the cost function value
            g (list): the vales of all constraints
            fail (0/1): indicates if the function finished successfully
        """
        # Cost and Constraints
        f = 0
        g = []
        fail = 0
        
        # Run building simulation
        x, cost, constraints = self.building.simulate(policy)
        f = np.sum(cost)
        g.append(np.sum(constraints))
        
#        print(f)
#        print(g[0])
        
        return f, g, fail
        
        
        
        
        
        
         
        
        
        

