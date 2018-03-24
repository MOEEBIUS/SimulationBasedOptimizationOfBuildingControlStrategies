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
from pyOpt import *

class Algorithm():
    """The high-level Algorithm Class. It defines an abstract optimization setup.
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
    """Utilization of the SLSQP algorithm of pyOpt package."""
     
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
             
        # Setupt Box Constrains in pyOpt
        self.opt_prob = Optimization('SLSQP Constrained Problem', self.wrapSimulation)
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
        
        self.opt(self.opt_prob, sens_step = 1e-6)
        print(self.opt_prob.solution(0))
        a =  self.opt_prob.solution(0)
        for ii in range(self.building.policy.shape[1]):
            self.building.policy[0, ii] = a.getVar(ii).value
        
        return self.building.policy
    
    def wrapSimulation(self, policy):
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
    """Utilization of the NSGA2 algorithm of pyOpt package."""
     
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
             
        # Setupt Box Constrains in pyOpt
        self.opt_prob = Optimization('NSGA2 Constrained Problem', self.wrapSimulation)
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
        
        self.opt(self.opt_prob)
        print(self.opt_prob.solution(0))
        a =  self.opt_prob.solution(0)
        for ii in range(self.building.policy.shape[1]):
            self.building.policy[0, ii] = a.getVar(ii).value
        
        return self.building.policy
    
    def wrapSimulation(self, policy):
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
        
        
        
         
        
        
        

