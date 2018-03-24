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

from Building import Building
import numpy as np

class SimpleRoomOpenLoop(Building):
    """The Class defining a Simple Room controlled by an open-loop controller.
    """
    def __init__(self):
        """Initialize all relevant parameters.
        """
        print("Initializing the Simple Room Open Loop Class...")
        # The state index in the simulation output array
        self.states = [0]
        # The actions index in the simulation output array
        self.actions = [2]
        # The disturbances (weather) index in the simulation output array
        self.disturbances = [1]
        # The simulation period
        self.T = 24
        # The initial (rule-based) controller
        self.policy = np.array([0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], ndmin = 2)
        # Initial exploration is activated
        self.rand = 0
        # Weather inputs (empty)
        self.w = []
        # Control parameter limits (min, max)
        self.controlLim = [0, 2]
        # Action limits (min, max)
        self.actionLim = [0, 2]
        # Thermal comfort limits (room temperature, (min, max))
        self.comfort = [20, 24]
        # Occupied interval (simulation time-steps)
        self.occ = [7, 17]
        # Number of optimization parameters
        self.nvars = 18
        # number of thermal comfort constraints
        self.ncons = 1
        
        
    def getConfiguration(self):
        return self.T, self.states, self.actions, self.disturbances, self.controlLim, self.actionLim, self.comfort, self.occ, self.nvars, self.ncons
    
    
    def setConfiguration(self, T, states, actions, disurbances, controlLim, actionLim, comfort, occ):
        self.T = T
        self.states = states
        self.actions = actions
        self.disurbances = disurbances
        self.controlLim = controlLim
        self.actionLim = actionLim
        self.comfort = comfort
        self.occ = occ
        
        
    def controller(self, policy, x, timeStep):
        """Generate control action(s) in each simulation time-step.
        Args:
            policy (numpy array): the controller to be evaluated
            x (numpy array): State at current time-step, used as input to the controller
            timeStep (float): The current time-step index

        Returns:
            u (numpy array): The control action(s)
        """

        # reshape policy vector to correct dimensions
        self.policy = np.array(policy).reshape(1, self.nvars)

        # Calculate action according to controller (one action per time-step)
        if(self.rand == 0):
            if((timeStep >= 0) and (timeStep <= 17)):
                u = self.policy[0, timeStep]
            else:
                u = 0
        
        # Quasi-random policy for initial guided exploration
        else:
            if((timeStep >= 0) and (timeStep <= 7)):
                u = 1 + np.random.randn() / 3.0
            if((timeStep > 7) and (timeStep <= 17)):
                u = 0.2 + np.random.randn() / 10.0
            if(timeStep > 17):
                u = 0
        
        # Project generated action within bounds
        if(u < self.actionLim[0]):
            u = self.actionLim[0]
        if(u > self.actionLim[1]):
            u = self.actionLim[1]
            
        return u
    
    
    def simulate(self, policy):
        
        # reshape policy vector to correct dimensions
        self.policy = np.array(policy).reshape(1, self.nvars)
        
        # Building parameters
        tau       = 2 * 3600
        Q0Hea     = 100
        UA        = Q0Hea / 20.0
        C         = 8 * tau*UA
        
        # Initial states (x_0)
        xx = np.zeros((self.T + 1, len(self.states) + 1))
        uu = np.zeros((self.T + 1, len(self.actions)))
        cc = np.zeros((self.T + 1, 1))
        xx[0,] = [17, self.w[0,]]
        uu[0,] = self.controller(self.policy, xx[0,], 0)
        cc[0,] = self.comfortConstraints(xx[0, self.states], 0)
        
        # Simulation
        for ii in range(1, self.T + 1):
            xx[ii,0] = xx[ii-1, 0] + 3600.0 / C * ( UA * (self.w[ii-1,] - xx[ii-1, 0] ) + Q0Hea * uu[ii-1,])
            xx[ii,1] = self.w[ii,]
            uu[ii,] = self.controller(self.policy, xx[ii,], ii)
            cc[ii,] = self.comfortConstraints(xx[ii, self.states], ii)
    
    
        # Gather states
        x = np.concatenate((xx, uu), axis=1)
       
        # Gather cost, constraints
        energy = uu.copy()
        energy = 3 * energy
        
        bill = uu.copy()
        for ii in range(0, uu.shape[0]):
            if(ii < 6):
                bill[ii, 0] = 5 * bill[ii, 0]
            else:
                bill[ii, 0] = 15 * bill[ii, 0]
 
#        print("Total Energy Consumption = " + str(np.sum(energy)))
#        print("Total Monetary Cost = " + str(np.sum(bill)))
#        print("Comfort Constraint = " + str(np.sum(cc)))
#        print("---------------------------------------------")
        
        # Return here either energy either bill, depending on Use Case
        return x, energy, cc
        
    
    def comfortConstraints(self, x, timeStep):
        """Calculate the thermal comfort constraints in each simulation time-step.
        Args:
            x (numpy array): State at current time-step, used as input to the controller
            timeStep (float): The current time-step index

        Returns:
            violations (float): A number indicating the total constraint violations
        """
        
        # Penalize thermal comfort (room temperature) violations
        violations = 0;
        for ii in range(0, x.shape[0]):
            if((timeStep >= self.occ[0]) and (timeStep <= self.occ[1])):
                if(x[ii,] < self.comfort[0]):
                    violations = violations + 10 * (x[ii,] - self.comfort[0])**2
                if(x[ii,] > self.comfort[1]):
                    violations = violations + 10 * (x[ii,] - self.comfort[1])**2
                    
        return violations
    
    
    
    
    