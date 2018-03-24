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
keepEnergyRB = []
keepComfortRB = []
keepEnergySLSQP = []
keepComfortSLSQP = []
keepEnergyGA = []
keepComfortGA = []

# Load results
baseResultsFolder = 'Results/Energy/'
for ii in range(1, days):
    
    # RB (Baseline) Results
    filename=baseResultsFolder + 'RBworkspace' + str(ii + 1) + '.out'
    with open(filename, 'rb') as f:
        policyRB, EnergyRB, ComfortRB, buildingRB = pickle.load(f)
    keepEnergyRB.append(EnergyRB[-1])
    keepComfortRB.append(ComfortRB[-1])
    
    # SLSQP Results
    filename=baseResultsFolder + 'SLSQPworkspace' + str(ii + 1) + '.out'
    with open(filename, 'rb') as f:
        policySLSQP, EnergySLSQP, ComfortSLSQP, buildingSLSQP = pickle.load(f)
    keepEnergySLSQP.append(EnergySLSQP[-1])
    keepComfortSLSQP.append(ComfortSLSQP[-1])
    
    # NSGA2 Results
    filename=baseResultsFolder + 'NSGA2workspace' + str(ii + 1) + '.out'
    with open(filename, 'rb') as f:
        policyGA, EnergyGA, ComfortGA, buildingGA = pickle.load(f)
    keepEnergyGA.append(EnergyGA[-1])
    keepComfortGA.append(ComfortGA[-1])
    

# Plot and log, Comfort Comparison
saveResultsFolder = 'Results/Comparisons/'
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(keepComfortRB, label='Comfort, RB')
ax.plot(keepComfortSLSQP, label='Comfort, SLSQP')
ax.plot(keepComfortGA, label='Comfort, NSGA2')
plt.title('Comfort Comparison of RB, SLSQP and GA under Energy Minimization Objective')
ax.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.show(fig)
art = []
art.append(ax.legend)
fig.savefig(saveResultsFolder + 'AllComfortComparisonEnergy.svg', format = 'svg', dpi = 1200, additional_artists = art, bbox_inches = "tight")
pickle.dump(fig, open(saveResultsFolder + 'AllComfortComparisonEnergy.figure', 'wb'))
plt.close(fig)

# Plot and log, Cost Comparison
saveResultsFolder = 'Results/Comparisons/'
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(keepEnergyRB, label='Consumption, RB')
ax.plot(keepEnergySLSQP, label='Consumption, SLSQP')
ax.plot(keepEnergyGA, label='Consumption, NSGA2')
plt.title('Consumption Comparison of RB, SLSQP and GA under Energy Minimization Objective')
ax.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.show(fig)
art = []
art.append(ax.legend)
fig.savefig(saveResultsFolder + 'AllConsumptionComparisonEnergy.svg', format = 'svg', dpi = 1200, additional_artists = art, bbox_inches = "tight")
pickle.dump(fig, open(saveResultsFolder + 'AllConsumptionComparisonEnergy.figure', 'wb'))
plt.close(fig)



