# thermoNETS

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/esa/thermonets">
    <img src="figures/thermonets.png" alt="Logo" width="280">
  </a>
  <p align="center">
    Thermosphere neural implicit representation
    <br />
    <a href="https://github.com/esa/thermonets/issues/new/choose">Report bug</a>
    ·
    <a href="https://github.com/esa/thermonets/issues/new/choose">Request feature</a>
  </p>
</p>

## Info

This repository introduces a novel neural architecture termed thermoNET, designed to represent thermospheric density in satellite orbital propagation, using a reduced amount of differentiable computations.

Due to the appearance of a neural network on the right-hand side of the equations of motion, the resulting satellite dynamics is governed by a NeuralODE, a neural Ordinary Differential Equation, characterized by its fully differentiable nature, allowing the derivation of variational equations (hence of the state transition matrix) and facilitating its use in connection to advanced numerical techniques such as Taylor-based numerical propagation and differential algebraic techniques. Efficient training of the network parameters occurs through two distinct approaches.

This repository contains the code to train, analyze and use thermoNETs for downstream tasks: including orbit propagation and neuralODE training. This was developed during the European Spage Agency's ACT study:

```
@inproceedings{thermonets,
  title = {NeuralODEs for VLEO simulations: Introducing thermoNET for Thermosphere Modeling},
  author = {Izzo, Dario and Acciarini, Giacomo and Biscani, Francesco},
  booktitle = {29th International Symposium on Space Flight Dynamics},
  year = {2024}
}
```
https://github.com/esa/thermonets/assets/3327087/09267ff1-4939-49a6-b8fd-be0cb26f2a60

## Goals
* represent thermospheric density empirical models via a lightweight neural network (only a few thousands parameters)
* neural ODE training to adjust trained architectures to match observed and/or simulated satellite data
* neural representation of NRLMSISE-00 and JB-08 (pre-trained) and available
* tutorial on the use of thermoNET for pure inference (i.e., as a thermospheric density model), during orbit propagation, for neural ODE fix (see `notebooks` folder)
https://github.com/esa/thermonets/assets/3327087/09267ff1-4939-49a6-b8fd-be0cb26f2a60

## Getting Started





