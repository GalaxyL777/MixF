# MixF
The Simulation Code of Propagation-induced Spectro-polarimetric Properties

The code file simulation.py contains two main parts: three classes and the simulation program.

1. The GFR (Generalized Faraday Rotation) class predicts the Q, U, and V behavior when a set of model parameters is given;
2. The PEPWs (Propagation Effects of the Polarized Wave) class predicts the Q, U, and V behavior when a bright polarized radio wave propagates through a weakly absorbing magnetized plasma;
3. The likelihood class of PEPWs. The main goal of this function is to construct a likelihood function with Gaussian noise, which can be directly used for maximum likelihood estimation and Bayesian inference.

The simulation begins with mock data generated from the GFR class, and then we consider the cold and hot plasma scenarios and fit the mock data. Finally, we plot the mock data and the best-fit model curves.
