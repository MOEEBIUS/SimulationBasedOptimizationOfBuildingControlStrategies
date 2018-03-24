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

""" Generate randon weather data for the Simple Room Example """
def generateRandomWeather(T, dt):
    TO = np.zeros((T + 1, 1))
    a = 5
    b = 13
    randomWeather1 = (b - a) * np.random.random_sample(1) + a #np.randnp.random.randint(a,b,1)
    randomWeather = a + (b - a) * np.random.random_sample()
    for ii in range(1, len(TO)): 
        TO[ii-1, 0] = randomWeather1 + randomWeather * np.abs(np.sin(0.00003 * ii * dt))
    TO[-1, 0] = TO[-2, 0]
    return TO

