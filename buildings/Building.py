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

class Building():
    """The high-level Building Class. It defines an abstract building with 
        arbitrary properties, systems and available controls.
    The main API methods are:
        simulate
        controller
        setConfiguration
        getConfiguration
    """

    def simulate(self):
        """Run a full episode of the building's dynamics.
        Args:
            policy (numpy array): the controller to be evaluated

        Returns:
            xx (numpy array): states
            cf (numpy array) : cost function per time-step
            cc (numpy array) : comfort constraints per time-step
        """
        raise NotImplementedError

    def controller(self, policy, x, timeStep):
        """Generates a control action(s) per simulation time-step.
        Args: 
            policy (numpy array): the controller
            x (numpy array): current states
            timeStep (float): current simulation time-step
            
        Returns: 
            u (numpy array): the control action(s).
        """
        raise NotImplementedError
        
    def setConfiguration(self, T, states, actions, disturbances, cotnrolLim, actionLim, comfort, occ):
        """Sets the specific Building Simulation propeties.
        Args:
            T (float): simulation period in time-steps
            states (list): index of states
            actions (list): index of actions
            disturbances (list): index of disturbances
            controlLim (list): control parameter bounds (min, max) or the optimization
            actionLim (list): control action bounds (min, max)
            comfort (list): thermal comfort bounds (min, max)
            occ (list): occupied period (time-steps)
        """
        raise NotImplementedError

    def getConfiguration(self):
        """Gets the specific Building Simulation propeties.
        Returns:
            T (float): simulation period in time-steps
            states (list): index of states
            actions (list): index of actions
            disturbances (list): index of disturbances
            controlLim (list): control parameter bounds (min, max) for the optimization
            actionLim (list): control action bounds (min, max)
            comfort (list): thermal comfort bounds (min, max)
            occ (list): occupied period (time-steps)
            nvars (float): number of optimization parameters
            ncons (float): number of thermal comfort constraints
        """
        raise NotImplementedError
        
        
        

