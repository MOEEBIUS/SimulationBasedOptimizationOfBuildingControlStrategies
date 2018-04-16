# Simulation-based Optimization of Building Control Strategies

We examine the problem of simulation-based optimization of building control strategies using detailed thermal simulation models (e.g. developed using EnergyPlus, TRNSYS or Modelica), as defined in [1, 2].

As the simulation time of these building models can be a significant bottleneck, we define an optimization approach that requires as less simulation calls as possible. To achieve this, we adopt ideas from sample-efficient algorithms developed within model-based Reinforcement Learning [3, 4] and data-driven control [5, 6] domains.


## References

 1. Kontes, G. D., Valmaseda, C., Giannakis, G. I., Katsigarakis, K. I., & Rovas, D. V. (2014). Intelligent BEMS design using detailed thermal simulation models and surrogate-based stochastic optimization. _Journal of Process Control, _24_(6), 846-855.
 2. Kontes, Georgios D. "Model Assisted Control for Energy Efficiency in Buildings." Ph.D. Thesis, Technical University of Crete, 2017.
 3. Deisenroth, M. P., and C. E. Rasmussen. "PILCO: A model-based and data-efficient approach to policy search." Proceedings of the 28th International Conference on Machine Learning. International Machine Learning Society, 2011.
 4. Deisenroth, Marc Peter, Dieter Fox, and Carl Edward Rasmussen. "Gaussian processes for data-efficient learning in robotics and control." IEEE Transactions on Pattern Analysis and Machine Intelligence 37.2 (2015): 408-423.
 5. Nghiem, Truong X., and Colin N. Jones. "Data-driven demand response modeling and control of buildings with gaussian processes." American Control Conference (ACC), 2017. IEEE, 2017.
 6. Jain, A., Nghiem, T. X., Morari, M., and Mangharam,, R. "Learning and control using Gaussian processes." Proceedings of the 9th ACM/IEEE International Conference on Cyber-Physical Systems, 2017.


## Dependencies

Requires PyOpt Library (http://www.pyopt.org/)


## License

This code is released by Technische Hochschule Nuernberg Georg Simon Ohm under the GNU General Public License version 3 (GPLv3) 


## Acknowledgments

Parts of this work have been developed with funding from the European Union's Horizon 2020 
research and innovation programme under grant agreement No 680517 (MOEEBIUS)