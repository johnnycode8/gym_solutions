<a name="readme-top"></a>

<h1 align="center">Gymnasium (Deep) Reinforcement Learning Tutorials</h1>

This repository contains a collection of Python code that solves/trains Reinforcement Learning environments from the [Gymnasium Library](https://gymnasium.farama.org/), formerly OpenAI’s Gym library. Each solution is accompanied by a video tutorial on my YouTube channel, [@johnnycode](https://www.youtube.com/@johnnycode), containing explanations and code walkthroughs. If you find the code and tutorials helpful, please consider supporting my work:

<a href='https://www.buymeacoffee.com/johnnycode'><img src="https://cdn.buymeacoffee.com/buttons/default-blue.png" alt="Buy Me A Coffee" height="41" width="174"></a>

<br />

# Train Atari Games <!-- omit from toc -->
If you want to jump straight into training AI agents to play Atari games, this tutorial requires no coding and no reinforcement learning experience! We use RL Baselines3 Zoo, a powerful training framework that lets you train and test AI models easily through a command line interface.

<a href='https://youtu.be/aQsaH7Tzvp0'><img src='https://img.youtube.com/vi/aQsaH7Tzvp0/0.jpg' width='400' alt='Full Guide: Easiest Way to Train AI to Play Atari Games with Deep Reinforcement Learning'/></a>

<br />

If you want to learn Reinforcement Learning:
- [Installation](#installation)
- [Beginner Reinforcement Learning Tutorials](#beginner-reinforcement-learning-tutorials)
    - [1. Q-Learning on Gymnasium FrozenLake-v1 (8x8 Tiles)](#1-q-learning-on-gymnasium-frozenlake-v1-8x8-tiles)
      - [Watch Q-Learning Values Change During Training on Gymnasium FrozenLake-v1](#watch-q-learning-values-change-during-training-on-gymnasium-frozenlake-v1)
    - [2. Q-Learning on Gymnasium Taxi-v3 (Multiple Objectives)](#2-q-learning-on-gymnasium-taxi-v3-multiple-objectives)
    - [3. Q-Learning on Gymnasium MountainCar-v0 (Continuous Observation Space)](#3-q-learning-on-gymnasium-mountaincar-v0-continuous-observation-space)
    - [4. Q-Learning on Gymnasium CartPole-v1 (Multiple Continuous Observation Spaces)](#4-q-learning-on-gymnasium-cartpole-v1-multiple-continuous-observation-spaces)
    - [5. Q-Learning on Gymnasium Acrobot-v1 (High Dimension Q-Table)](#5-q-learning-on-gymnasium-acrobot-v1-high-dimension-q-table)
    - [6. Q-Learning on Gymnasium Pendulum-v1 (Continuous Action and Observation Spaces)](#6-q-learning-on-gymnasium-pendulum-v1-continuous-action-and-observation-spaces)
    - [7. Q-Learning on Gymnasium MountainCarContinuous-v0 (Stuck in Local Optima)](#7-q-learning-on-gymnasium-mountaincarcontinuous-v0-stuck-in-local-optima)
- [Deep Reinforcement Learning Tutorials](#deep-reinforcement-learning-tutorials)
    - [Getting Started with Neural Networks](#getting-started-with-neural-networks)
    - [Deep Q-Learning a.k.a Deep Q-Network (DQN) Explained](#deep-q-learning-aka-deep-q-network-dqn-explained)
      - [Implement DQN with PyTorch and Train Flappy Bird](#implement-dqn-with-pytorch-and-train-flappy-bird)
    - [Apply DQN to Gymnasium Mountain Car](#apply-dqn-to-gymnasium-mountain-car)
    - [Get Started with Convolutional Neural Network (CNN)](#get-started-with-convolutional-neural-network-cnn)
- [Stable Baselines3 Tutorials](#stable-baselines3-tutorials)
    - [Stable Baselines3: Get Started Guide | Train Gymnasium MuJoCo Humanoid-v4](#stable-baselines3-get-started-guide--train-gymnasium-mujoco-humanoid-v4)
    - [Stable Baselines3 - Beginner's Guide to Choosing RL Algorithms for Training](#stable-baselines3---beginners-guide-to-choosing-rl-algorithms-for-training)
    - [Stable Baselines3: Dynamically Load RL Algorithm for Training | Train Gymnasium Pendulum](#stable-baselines3-dynamically-load-rl-algorithm-for-training--train-gymnasium-pendulum)
    - [Automatically Stop Training When Best Model is Found in Stable Baselines3](#automatically-stop-training-when-best-model-is-found-in-stable-baselines3)

<br />

# Installation
The [Gymnasium Library](https://gymnasium.farama.org/) is supported on Linux and Mac OS, but not officially on Windows. On Windows, the Box2D package (Bipedal Walker, Car Racing, Lunar Lander) is problematic during installation, you may see errors such as:
* ERROR: Failed building wheels for box2d-py
* ERROR: Command swig.exe failed
* ERROR: Microsoft Visual C++ 14.0 or greater is required.

My Gymnasium on Windows installation guide shows how to resolve these errors and successfully install the complete set of Gymnasium Reinforcement Learning environments:

<a href='https://youtu.be/gMgj4pSHLww&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/gMgj4pSHLww/0.jpg' width='400' alt='How to Install Gymnasium on Windows'/></a>

However, due to the constantly evolving nature of software versions, you might still encounter issues with the above guide. As an alternative, you can install Gymnasium on Linux within Windows, using Windows Subsystem for Linux (WSL). In this guide, I show how to install the Gymnasium Box2D environments (Bipedal Walker, Car Racing, Lunar Lander) onto WSL:

<a href='https://youtu.be/yxS5WErjYxc&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/yxS5WErjYxc/0.jpg' width='400' alt='Install Gymnasium Box2D on Windows Subsystem for Linux'/></a>


<br />

# Beginner Reinforcement Learning Tutorials

### 1. Q-Learning on Gymnasium FrozenLake-v1 (8x8 Tiles)
This is the recommended starting point for beginners. This Q-Learning tutorial provides a step-by-step walkthrough of the code to solve the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 8x8 map. The Frozen Lake environment is simple and straightforward, allowing us to concentrate on understanding how Q-Learning works. The Epsilon-Greedy algorithm is used for both exploration (choosing random actions) and exploitation (choosing the best actions). Please note that this tutorial does not delve into the theory or math behind Q-Learning; it is purely focused on practical application.

<a href='https://youtu.be/ZhoIgo3qqLU&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/ZhoIgo3qqLU/0.jpg' width='400' alt='How to Train Gymnasium FrozenLake-v1 with Q-Learning'/></a>

##### Code Reference: <!-- omit from toc -->
* [frozen_lake_q.py](https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_q.py)

#### Watch Q-Learning Values Change During Training on Gymnasium FrozenLake-v1
This is the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment "enhanced" to help you better understand Q-Learning. The Q-values are overlaid on top of each cell of the map, allowing you to visually see the Q-values update in real-time during training. The map is enlarged to fill the entire screen, making the overlaid Q-values easier to read. Additionally, shortcut keys are available to speed up or slow down the animation.

<a href='https://youtu.be/1W_LOB-0IEY&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/1W_LOB-0IEY/0.jpg' width='400' alt='See Q-Learning in Realtime on FrozenLake-v1'/></a>

##### Code Reference: <!-- omit from toc -->
* [frozen_lake_enhanced.py](https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_enhanced.py)
This is the FrozenLake-v1 environment overlayed with Q values. You do not need to understand this code, but feel free to check how I modified the environment.
* [frozen_lake_qe.py](https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_qe.py)
This file is almost identical to the frozen_lake_q.py file above, except this uses the frozen_lake_enhanced.py environment.


<br />


### 2. Q-Learning on Gymnasium Taxi-v3 (Multiple Objectives)
In the [Taxi-V3](https://gymnasium.farama.org/environments/toy_text/taxi/) environment, the agent (Taxi) learns to pick up passengers and deliver them to their destination. It is very much similar to the Frozen Lake environment, except that the observation space is more complicated.

<a href='https://youtu.be/9fAnzZ6xzhA&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/9fAnzZ6xzhA/0.jpg' width='400' alt='How to Train Gymnasium Taxi-v3 Q-Learning'/></a>

##### Code Reference: <!-- omit from toc -->
* [taxi_q.py](https://github.com/johnnycode8/gym_solutions/blob/main/taxi_q.py)


<br />


### 3. Q-Learning on Gymnasium MountainCar-v0 (Continuous Observation Space)
This Q-Learning tutorial solves the [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/) environment. It builds upon the code from the Frozen Lake environment. What is interesting about this environment is that the observation space is continuous, whereas the Frozen Lake environment's observation space is discrete. "Discrete" means that the agent, the elf in Frozen Lake, steps from one cell on the grid to the next, so there is a clear distinction that the agent is going from one state to another. "Continuous" means that the agent, the car in Mountain Car, traverses the mountain on a continuous road, with no clear distinction of states.

<a href='https://youtu.be/_SWnNhM5w-g&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/_SWnNhM5w-g/0.jpg' width='400' alt='How to Train Gymnasium MountainCar-v0 with Q-Learning'/></a>

##### Code Reference: <!-- omit from toc -->
* [mountain_car_q.py](https://github.com/johnnycode8/gym_solutions/blob/main/mountain_car_q.py)


<br />


### 4. Q-Learning on Gymnasium CartPole-v1 (Multiple Continuous Observation Spaces)
This Q-Learning tutorial solves the [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment. It builds upon the code from the Frozen Lake environment. Like Mountain Car, the Cart Pole environment's observation space is also continuous. However, it has a more complicated continuous observation space: the cart's position and velocity and the pole's angle and angular velocity.

<a href='https://youtu.be/2u1REHeHMrg&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/2u1REHeHMrg/0.jpg' width='400' alt='How to Train Gymnasium CartPole-v1 with Q-Learning'/></a>

##### Code Reference: <!-- omit from toc -->
* [cartpole_q.py](https://github.com/johnnycode8/gym_solutions/blob/main/cartpole_q.py)


<br />


### 5. Q-Learning on Gymnasium Acrobot-v1 (High Dimension Q-Table)
We'll use a 7-dimension Q-Table to solve the [Acrobot-v1](https://gymnasium.farama.org/environments/classic_control/acrobot/) environment.

<a href='https://youtu.be/Pf1lEv3b5s4&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/Pf1lEv3b5s4/0.jpg' width='400' alt='How to Train Gymnasium Acrobot-v1 with Q-Learning'/></a>

##### Code Reference: <!-- omit from toc -->
* [acrobot_q.py](https://github.com/johnnycode8/gym_solutions/blob/main/acrobot_q.py)


<br />


### 6. Q-Learning on Gymnasium Pendulum-v1 (Continuous Action and Observation Spaces)
We'll use Q-Learning to solve the [Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/) environment.

<a href='https://youtu.be/o2NMWV5sImM&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/o2NMWV5sImM/0.jpg' width='400' alt='How to Train Gymnasium Pendulum-v1 with Q-Learning'/></a>

##### Code Reference: <!-- omit from toc -->
* [pendulum_q.py](https://github.com/johnnycode8/gym_solutions/blob/main/pendulum_q.py)


<br />


### 7. Q-Learning on Gymnasium MountainCarContinuous-v0 (Stuck in Local Optima)
We'll use Q-Learning to solve the [MountainCarContinuous-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) environment.

<a href='https://youtu.be/1Ms2UqRC8LA&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/1Ms2UqRC8LA/0.jpg' width='400' alt='How to Train Gymnasium MountainCarContinuous-v0 with Q-Learning'/></a>

##### Code Reference: <!-- omit from toc -->
* [mountain_car_cont_q.py](https://github.com/johnnycode8/gym_solutions/blob/main/mountain_car_cont_q.py)


<br /><br />



# Deep Reinforcement Learning Tutorials

### Getting Started with Neural Networks
Before diving into Deep Reinforcement Learning, it would be helpful to have a basic understanding of Neural Networks. This hands-on end-to-end example of how to calculate Loss and Gradient Descent on the smallest network.

<a href='https://youtu.be/6kOvmZDEMdc&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/6kOvmZDEMdc/0.jpg' width='400' alt='Work Thru the Most Basic Neural Network with Simplified Math and Python'/></a>

##### Code Reference: <!-- omit from toc -->
* [Basic Neural Network](https://github.com/johnnycode8/basic_neural_network) repo

<br />

### Deep Q-Learning a.k.a Deep Q-Network (DQN) Explained
This Deep Reinforcement Learning tutorial explains how the Deep Q-Learning (DQL) algorithm uses two neural networks: a Policy Deep Q-Network (DQN) and a Target DQN, to train the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 4x4 environment. The Frozen Lake environment is very simple and straightforward, allowing us to focus on how DQL works. The Epsilon-Greedy algorithm and the Experience Replay technique are also used as part of DQL to help train the learning agent. The code referenced here is also walked through in the YouTube tutorial. PyTorch is used to build the DQNs.

YouTube Tutorial Content:
* Quick overview of the Frozen Lake environment.
* Why use Reinforcement Learning on Frozen Lake, if a simple search algorithm works.
* Overview of the Epsilon-Greedy algorithm.
* Compare Q-Learning's Q-Table vs Deep Q-Learning's DQN
* How the Q-Table learns.
* How the DQN learns.
* Overview of Experience Replay.
* Putting it all together - walkthru of the Deep Q-Learning algorithm.
* Walkthru of the Deep Q-Learning code for Frozen Lake.
* Run and demo the training code.

<a href='https://youtu.be/EUrWGTCGzlA&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/EUrWGTCGzlA/0.jpg' width='400' alt='Deep Q-Learning DQL/DQN Explained + Code Walkthru + Demo'/></a>

##### Code Reference: <!-- omit from toc -->
* [frozen_lake_dql.py](https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py)

##### Dependencies: <!-- omit from toc -->
* <a href='https://pytorch.org/'>PyTorch</a>

#### Implement DQN with PyTorch and Train Flappy Bird
To gain in-depth understanding of the DQN algorithm, try my series on implementing DQN from scratch: [DQN PyTorch Beginner Tutorials] (https://github.com/johnnycode8/dqn_pytorch)


<br />

### Apply DQN to Gymnasium Mountain Car
We've already solve [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/) with Q-Learning (above). For learning purposes, we'll do it again with Deep Q-Learning. Hopefully, it'll give you a better understanding on how to adapt the code for your needs.

<a href='https://youtu.be/oceguqZxjn4&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/oceguqZxjn4/0.jpg' width='400' alt='How to Traing Gymnasium MountainCar-V0 with Deep Q-Learning'/></a>

##### Code Reference: <!-- omit from toc -->
* [mountain_car_dql.py](https://github.com/johnnycode8/gym_solutions/blob/main/mountain_car_dql.py)

##### Dependencies: <!-- omit from toc -->
* <a href='https://pytorch.org/'>PyTorch</a>

<br />

### Get Started with Convolutional Neural Network (CNN)
In part 1 (above), the Deep Q-Networks (DQN) used were straightforward neural networks with a hidden layer and an output layer. This network architecture works for simple environments. However, for complex environments—such as Atari Pong—where the agent learns from the environment visually, we need to modify our DQNs with convolutional layers. We'll continue the explanation on the very simple [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 4x4 environment, however, we'll modify the inputs such that they are treated as images.

<a href='https://youtu.be/qKePPepISiA&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/qKePPepISiA/0.jpg' width='400' alt='Deep Q-Learning with Convolutional Neural Networks'/></a>

##### Code Reference: <!-- omit from toc -->
* [frozen_lake_dql_cnn.py](https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql_cnn.py)

##### Dependencies: <!-- omit from toc -->
* <a href='https://pytorch.org/'>PyTorch</a>

<br /><br />

# Stable Baselines3 Tutorials

### Stable Baselines3: Get Started Guide | Train Gymnasium MuJoCo Humanoid-v4
Get started with the Stable Baselines3 Reinforcement Learning library by training the Gymnasium MuJoCo [Humanoid-v4](https://gymnasium.farama.org/environments/mujoco/humanoid/) environment with the Soft Actor-Critic (SAC) algorithm. The focus is on the usage of the Stable Baselines3 (SB3) library and the use of TensorBoard to monitor training progress. Other algorithms used in the demo include Twin Delayed Deep Deterministic Policy Gradient (TD3) and Advantage Actor Critic (A2C).

<a href='https://youtu.be/OqvXHi_QtT0&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/OqvXHi_QtT0/0.jpg' width='400' alt='How to Train Gymnasium Humanoid-v4 with Stable Baselines3'/></a>

##### Code Reference: <!-- omit from toc -->
* [sb3.py](https://github.com/johnnycode8/gym_solutions/blob/main/sb3.py)

##### Dependency: <!-- omit from toc -->
* [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/)

<br />

### Stable Baselines3 - Beginner's Guide to Choosing RL Algorithms for Training
SB3 offers many ready-to-use RL algorithms out of the box, but as beginners, how do we know which algorithms to use? We'll discuss this topic in the video:

<a href='https://youtu.be/2AFl-iWGQzc&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/2AFl-iWGQzc/0.jpg' width='400' alt='Beginners Guide on Choosing Stable Baselines3 Algorithms for Training'/></a>

<br />

### Stable Baselines3: Dynamically Load RL Algorithm for Training | Train Gymnasium Pendulum
In part 1, for simplicity, the algorithms (SAC, TD3, 2C) were hardcoded in the code. In part 2, we'll make loading and creating instances of the algorithms dynamic. To test the changes, we'll train [Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/) using SAC and TD3 simultaneously and monitor the progress thru TensorBoard.

<a href='https://youtu.be/nf2IE2GEJ-s&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/nf2IE2GEJ-s/0.jpg' width='400' alt='How to Train Gymnasium Pendulum-v1 with Stable Baselines3'/></a>

##### Code Reference: <!-- omit from toc -->
* [sb3v2.py](https://github.com/johnnycode8/gym_solutions/blob/main/sb3v2.py)

<br />

### Automatically Stop Training When Best Model is Found in Stable Baselines3
This tutorial walks thru the code that automatically stop training when the best model is found. We'll demonstrate by training the Gymnasium [BipedalWalker-v3](https://gymnasium.farama.org/environments/box2d/bipedal_walker/) using Soft-Actor Critic.

<a href='https://youtu.be/mCkgLweyMqo&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/mCkgLweyMqo/0.jpg' width='400' alt='How to Train Gymnasium BipedalWalker-v3 with Stable Baselines3'/></a>

##### Code Reference: <!-- omit from toc -->
* [sb3v3.py](https://github.com/johnnycode8/gym_solutions/blob/main/sb3v3.py)


<p align="right">(<a href="#readme-top">back to top</a>)</p>
