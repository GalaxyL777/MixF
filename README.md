# MixF
The Simulation Code of Propagation-induced Spectro-polarimetric Properties

The code file Faraday_conversion.py contains two main parts: three types of class and the simulation program.

1. The GFR (Generalized Faraday Rotation) class predicts the Q, U, and V behavior when a set of model parameters is given;
2. The conversion and conversions (no absorption scenario) classes predict the Q, U, and V behavior when a bright polarized radio wave propagates through a magnetized plasma. In conversions_class, we consider a more realistic model in which the magnetized plasma consists of three layers: the foreground medium, the conversion medium, and the background medium;
3. The likelihood class of Faraday conversion. The main goal of this function is to construct a likelihood function with Gaussian noise, which can be directly used for maximum likelihood estimation and Bayesian inference.

The simulation begins with mock data generated from the GFR class, and then we consider the cold and hot plasma scenarios and fit the mock data. Finally, we plot the mock data and the best-fit model curves.

In conversion_1124A2.py, we employ our model to burst 926 of FRB 20201124A2 (Xu et al. 2022).
