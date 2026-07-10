# FLARE

<p align="center">
  <img src="FLARE/figures/fig0.svg" alt="FLARE framework" width="100%">
</p>

**A Forced Latent Autoencoder for Response Equations**

## Paper

**Discovering Latent Response Laws in Forced Physical Systems**

## Abstract

Governing equations provide compact descriptions of physical systems, yet the variables in which they are simple are often hidden in high-dimensional measurements. This challenge is sharper for forced systems, whose responses depend on both intrinsic dynamics and time-dependent inputs.
Here we introduce FLARE, a forced latent autoencoder for response equations that learns compact response coordinates, identifies sparse input-dependent latent dynamics and decodes equation rollouts to full responses. By estimating latent dimension from data and separating state estimation from external forcing, FLARE enables forecasts to be initialized from past responses and driven by prescribed future inputs.
Across known dynamical systems, application-scale forced responses and visual observations, FLARE recovers compact forced dynamics and predicts long-horizon high-dimensional responses under inputs not used for training. By turning learned coordinates into a dynamical interface, FLARE extends equation discovery to systems whose effective states are hidden within complex observations, providing a route for interpretable modelling and prediction of high-dimensional responses in forced dynamical systems.

## Repository Contents

This repository currently contains:

- result files;
- dataset generation file;
- figure and visualization files.

The source code and detailed implementation will be released upon acceptance of the manuscript.
