# ECLAIR RL Environments

Welcome to ECLAIR's repository of RL and control algorithms. This is where we keep all of our algorithms to control our robots.

# File Organization

* `controlAlgorithms`:
* `customEnvs`: custom OpenAI gym environments
* `media`: any images or videos
* `tests`: executable test files for our algorithms
* `urdfs`: URDF files which contain physics and joint information for robots and props
* `utils`: custom data structures and utility functions 

# Dependencies

## Basics

These are just the dependencies required to run a typical ECLAIR robotics software project. You may have to install additional packages for specific simulation engines. 

You can install the dependencies through either through pip or conda.

* matplotlib
* pytorch
* gym

## Robotic Simulaiton Packages (updated as of 2022)

Different robotics software simulation packages have different advantages and disadvantages which you should research up on before deciding on a simulaton engine for your project. We are currently using Bullet. Here are a few of simulation engines that have been the standard for a long time:

* **Bullet**: general rigid/soft body dynamics originally for games
* **Mujoco** (most used in robotics research today): soft body contact modeling, muscle tendon modeling
* **ODE**: General rigid body dynamics originally made for games
* **PhysX**: Developed by NVIDIA for games. It is used by Unreal and Unreal engine. Recently, it is also being used in Nvidia's Omniverse framework.

Here are relatively new but stable simulation engines:

* **Raisim**: Developed at ETH Zurich and showed success at latest DARPA Underground Challenge
* **Drake**: Developed by a lab at MIT for hand manipulation robots and boasts accurate reflection of 