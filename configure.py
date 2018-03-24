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

import sys, os

""" Add necessary files to python path """

scriptPath = os.getcwd()
sys.path.append(scriptPath + '\\algorithms\\')
sys.path.append(scriptPath + '\\buildings\\')
sys.path.append(scriptPath + '\\buildings\\SimpleRoomOpenLoop\\')
sys.path.append(scriptPath + '\\experiments\\')
sys.path.append(scriptPath + '\\experiments\\TestSimpleRoomOpenLoop\\')
