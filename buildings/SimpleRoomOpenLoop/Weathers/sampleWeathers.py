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

import pickle
from generateRandomWeather import generateRandomWeather
dt = 3600.0
T = 24

""" Generate randon weather data for the Simple Room Example """
for ii in range(0, 366 * 2):
    w = generateRandomWeather(T, dt)
    
    # Save all variables to a file
    filename='w' + str(ii) + '.out'
    
    with open(filename, 'w') as f:
        pickle.dump(w, f)
    

