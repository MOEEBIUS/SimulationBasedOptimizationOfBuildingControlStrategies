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

import numpy as np
import pickle
from pylab import *

import sys

# Number of simulated days
days = 10

# Logging of values
keepCostRB = []
keepComfortRB = []
keepCostSLSQP = []
keepComfortSLSQP = []
keepCostGA = []
keepComfortGA = []

# Load results
baseResultsFolder = 'Results/Bill/'
for ii in range(0, days):
    
    # RB (Baseline) Results
    filename=baseResultsFolder + 'RBworkspace' + str(ii + 1) + '.out'
    with open(filename, 'rb') as f:
        policyRB, CostRB, ComfortRB, buildingRB = pickle.load(f)
    keepCostRB.append(CostRB[-1])
    keepComfortRB.append(ComfortRB[-1])
    
    # SLSQP Results
    filename=baseResultsFolder + 'SLSQPworkspace' + str(ii + 1) + '.out'
    with open(filename, 'rb') as f:
        policySLSQP, CostSLSQP, ComfortSLSQP, buildingSLSQP = pickle.load(f)
    keepCostSLSQP.append(CostSLSQP[-1])
    keepComfortSLSQP.append(ComfortSLSQP[-1])
    
    # NSGA2 Results
    filename=baseResultsFolder + 'NSGA2workspace' + str(ii + 1) + '.out'
    with open(filename, 'rb') as f:
        policyGA, CostGA, ComfortGA, buildingGA = pickle.load(f)
    keepCostGA.append(CostGA[-1])
    keepComfortGA.append(ComfortGA[-1])
    

# Plot and log, Comfort Comparison
saveResultsFolder = 'Results/Comparisons/'
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(keepComfortRB, label='Comfort, RB')
ax.plot(keepComfortSLSQP, label='Comfort, SLSQP')
ax.plot(keepComfortGA, label='Comfort, NSGA2')
plt.title('Comfort Comparison of RB, SLSQP and GA under Cost Minimization Objective')
ax.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.show(fig)
art = []
art.append(ax.legend)
fig.savefig(saveResultsFolder + 'AllComfortComparisonCost.svg', format = 'svg', dpi = 1200, additional_artists = art, bbox_inches = "tight")
pickle.dump(fig, open(saveResultsFolder + 'AllComfortComparisonCost.figure', 'wb'))
plt.close(fig)

# Plot and log, Cost Comparison
saveResultsFolder = 'Results/Comparisons/'
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(keepCostRB, label='Monetary Cost, RB')
ax.plot(keepCostSLSQP, label='Monetary Cost, SLSQP')
ax.plot(keepCostGA, label='Monetary Cost, NSGA2')
plt.title('Monetary Cost Comparison of RB, SLSQP and GA under Cost Minimization Objective')
ax.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.show(fig)
art = []
art.append(ax.legend)
fig.savefig(saveResultsFolder + 'AllBillComparisonCost.svg', format = 'svg', dpi = 1200, additional_artists = art, bbox_inches = "tight")
pickle.dump(fig, open(saveResultsFolder + 'AllBillComparisonCost.figure', 'wb'))
plt.close(fig)



