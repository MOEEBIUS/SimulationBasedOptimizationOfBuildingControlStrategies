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

import time
import pickle
from pylab import *
import numpy as np

from SimpleRoomOpenLoop.SimpleRoomOpenLoop import SimpleRoomOpenLoop
from Algorithms import SLSQP_pyOpt, NSGA2_pyOpt



def logResults(baseFolder, prefix, building, policy, costFunction, keepCost, keepComfort):
    """ Plot and logging helper function """
    
    # Plot relevant metrics
    xx, cf, cc = building.simulate(policy)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(building.w, label='Weather')
    ax.plot(xx[:,building.states], label='Room Temperature')
    ax.plot(xx[:,building.actions], label='Control Actions')
    ax.plot([1, building.T], [building.comfort[0], building.comfort[0]])
    ax.plot([1, building.T], [building.comfort[1], building.comfort[1]])
    ax.plot([building.occ[0], building.occ[0]], [building.comfort[0], building.comfort[1]])
    ax.plot([building.occ[1], building.occ[1]], [building.comfort[0], building.comfort[1]])
    if(costFunction ==0):
        ax.plot(cf, label='Energy Consumption')
    else:
        ax.plot(cf, label='Energy Bill')
    plt.title('Constraints satisfaction')
    ax.legend()
    plt.draw()
    fig.savefig(baseFolder + prefix + 'Constraints' + str(ii + 1) + '.svg', format = 'svg', dpi = 1200)
    pickle.dump(fig, open(baseFolder + prefix + 'Constraints' + str(ii + 1) + '.figure', 'wb')) 
    plt.close(fig)
    
    fig = plt.figure()
    ax = plt.subplot(111)
    if(costFunction ==0):
        ax.plot(keepCost, label='Energy Consumption')
    else:
        ax.plot(keepCost, label='Energy Bill')
    ax.plot(keepComfort, label='Comfort')
    plt.title('Convergence')
    ax.legend()
    plt.draw()
    fig.savefig(baseFolder + prefix + 'CostComfort' + str(ii + 1)+'.svg', format = 'svg', dpi = 1200)
    plt.close(fig)
    
    return xx, cf, cc



""" Main program """
# Choose which optimizers to test
flagSLSQP = 1
flagNSGA2 = 1
flagRB = 1

# Define a folder for storing the results
costFunction = 0 # 0: minEnergy; 1: Tariff;
if(costFunction == 0):
    baseFolder = 'Results/Energy/'
elif(costFunction == 1):
    baseFolder = 'Results/Bill/'

# Define the building
building = SimpleRoomOpenLoop()

# Extract the Rule-Based (Baseline) policy of the building
policyRB = building.policy

# Optimization results
keepCostRB = []
keepComfortRB = []
keepCostSLSQP = []
keepComfortSLSQP = []
keepCostNSGA2 = []
keepComfortNSGA2 = []

# Loop for 10 sample days
for ii in range(10):
    t0 = time.time()
    
    # Load weather
    weather = ii + 1
    filename='../../buildings/SimpleRoomOpenLoop/Weathers/w'+str(weather)+'.out'
    # Lading the objects:
    with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
        w = pickle.load(f)
    print('Using weather file: '+ 'Weathers/w'+str(weather)+'.out')
    
    # Setup weather conditions for the Simulation/Optimization 
    building.w = w.copy()

    if(flagSLSQP == 1):
        # Setup and configure the SLSQP Optimizer for the defined building
        optimizerSLSQP = SLSQP_pyOpt()
        ppSLSQP = optimizerSLSQP.configure(building)
        options = {'ACC': 1.0e-20, 'MAXIT': 2000, 'IPRINT': 1}
        resSLSQP = optimizerSLSQP.optimize(options)

    if(flagNSGA2 == 1):
        # Setup and configure the NSGA2 Optimizer for the defined building
        optimizerNSGA2 = NSGA2_pyOpt()
        ppNSGA2 = optimizerNSGA2.configure(building)
        options = {'PopSize': 360, 'maxGen': 2000, 'PrintOut' :2}
        resNSGA2 = optimizerNSGA2.optimize(options)

    
    # Logging and plotting data
    print("--------------------------------------------------")
    print("Optimization Results for Day "+ str(ii + 1) + ":")
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    
    
    if(flagRB == 1):
        print("--------------------------------------------------")
        dummyOptRB = SLSQP_pyOpt()
        dummyResultRB = dummyOptRB.configure(building)
        RB = dummyOptRB.wrapSimulation(policyRB)
        print("Performance of the baseline controller: " + str(RB))
        xx, cf, cc = logResults(baseFolder, 'RB', building, policyRB, costFunction, keepCostRB, keepComfortRB)
        keepCostRB.append(np.sum(cf))
        keepComfortRB.append(np.sum(cc))
        
        # Save all variables to a file 
        filename= baseFolder + 'RB' + 'workspace' + str(ii + 1) + '.out'
        # Saving the objects:
        with open(filename, 'wb') as f:
            pickle.dump([policyRB, keepCostRB, keepComfortRB, building], f)
        
    if(flagSLSQP == 1):
        
        print("--------------------------------------------------")
        pSLSQP = optimizerSLSQP.wrapSimulation(resSLSQP)
        print("Performance of the SLSQP controller: " + str(pSLSQP))
        xx, cf, cc = logResults(baseFolder, 'SLSQP', building, resSLSQP, costFunction, keepCostSLSQP, keepComfortSLSQP)
        keepCostSLSQP.append(np.sum(cf))
        keepComfortSLSQP.append(np.sum(cc))
        
        # Save all variables to a file 
        filename= baseFolder + 'SLSQP' + 'workspace' + str(ii + 1) + '.out'
        # Saving the objects:
        with open(filename, 'wb') as f:
            pickle.dump([resSLSQP, keepCostSLSQP, keepComfortSLSQP, building], f)
    
    if(flagNSGA2 == 1):
        print("--------------------------------------------------")
        pNSGA2 = optimizerNSGA2.wrapSimulation(resNSGA2)
        print("Performance of the NSGA2 controller: " + str(pNSGA2))
        xx, cf, cc = logResults(baseFolder, 'NSGA2', building, resNSGA2, costFunction, keepCostNSGA2, keepComfortNSGA2)
        keepCostNSGA2.append(np.sum(cf))
        keepComfortNSGA2.append(np.sum(cc))
        
        # Save all variables to a file 
        filename= baseFolder + 'NSGA2' + 'workspace' + str(ii + 1) + '.out'
        # Saving the objects:
        with open(filename, 'wb') as f:
            pickle.dump([resNSGA2, keepCostNSGA2, keepComfortNSGA2, building], f)
        
        
    elapsed = (time.time() - t0)
    print("Execution Time = " + str(elapsed) + " seconds...")
        
        


