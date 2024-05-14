<a name="readme-top"></a>

<h2 align="center">Gymnasium (Deep) Reinforcement Learning Tutorials</h2>

Collection of Python code that solves/trains Reinforcement Learning environments from the [Gymnasium Library](https://gymnasium.farama.org/), formerly OpenAI’s Gym library. Each solution has a companion video explanation and code walkthrough from my YouTube channel [@johnnycode](https://www.youtube.com/@johnnycode). If the code and video helped you, please consider:  
<a href='https://www.buymeacoffee.com/johnnycode'><img src="https://cdn.buymeacoffee.com/buttons/default-blue.png" alt="Buy Me A Coffee" height="41" width="174"></a>

## Installation
The [Gymnasium Library](https://gymnasium.farama.org/) is supported on Linux and Mac OS, but not officially on Windows. On Windows, the Box2D package (Bipedal Walker, Car Racing, Lunar Lander) is problematic during installation, you may see errors such as:
* ERROR: Failed building wheels for box2d-py
* ERROR: Command swig.exe failed
* ERROR: Microsoft Visual C++ 14.0 or greater is required.

My Gymnasium on Windows installation video shows you how to resolve these errors and successfully install the complete set of Gymnasium Reinforcement Learning environments.

##### YouTube Tutorial:
<a href='https://youtu.be/gMgj4pSHLww&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/gMgj4pSHLww/0.jpg' width='400' alt='Install Gymnasium on Windows'/></a>

<br />

# Beginner Reinforcement Learning Tutorials

## Q-Learning - Frozen Lake 8x8
This is the recommended starting point for beginners. This Q-Learning tutorial walks through the code on how to solve the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 8x8 map. The Frozen Lake environment is very simple and straightforward, allowing us to focus on how Q-Learning works. The Epsilon-Greedy algorithm is also used in conjunction with Q-Learning. Note that this tutorial does not explain the theory or math behind Q-Learning. 

##### Code Reference:
* [frozen_lake_q.py](https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_q.py) 

##### YouTube Tutorial:
<a href='https://youtu.be/ZhoIgo3qqLU&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/ZhoIgo3qqLU/0.jpg' width='400' alt='Solve FrozenLake-v1 8x8 with Q-Learning'/></a>

<br />

## Q-Learning - Frozen Lake 8x8 Enhanced
This is the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment "enhanced" to help you better understand Q-Learning. Features:
* The Q values are overlayed on top of each cell of the map, so that you can visually see the Q values update in realtime while training!
* The map is enlarged to fill the whole screen so that it is easier to read the overlayed Q values.
* Shortcut keys to speed up or slow down the animation.

##### Code Reference:
* [frozen_lake_qe.py](https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_qe.py)  
This file is almost identical to the frozen_lake_q.py file above, except this uses the frozen_lake_enhanced.py environment. 
* [frozen_lake_enhanced.py](https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_enhanced.py)  
This is the FrozenLake-v1 environment overlayed with Q values. You do not need to understand this code, but feel free to check how I modified the environment.

##### YouTube Tutorial:
<a href='https://youtu.be/1W_LOB-0IEY&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/1W_LOB-0IEY/0.jpg' width='400' alt='See Q-Learning in Realtime on FrozenLake-v1'/></a>

<br />

## Q-Learning - Mountain Car - Continuous Observation Space
This Q-Learning tutorial solves the [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/) environment. It builds upon the code from the Frozen Lake environment. What is interesting about this environment is that the observation space is continuous, whereas the Frozen Lake environment's observation space is discrete. "Discrete" means that the agent, the elf in Frozen Lake, steps from one cell on the grid to the next, so there is a clear distinction that the agent is going from one state to another. "Continuous" means that the agent, the car in Mountain Car, traverses the mountain on a continuous road, with no clear distinction of states.

##### Code Reference:
* [mountain_car_q.py](https://github.com/johnnycode8/gym_solutions/blob/main/mountain_car_q.py) 

##### YouTube Tutorial:
<a href='https://youtu.be/_SWnNhM5w-g&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/_SWnNhM5w-g/0.jpg' width='400' alt='Solves the MountainCar-v0 with Q-Learning'/></a>

<br />

## Q-Learning - Cart Pole - Multiple Continuous Observation Spaces
This Q-Learning tutorial solves the [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment. It builds upon the code from the Frozen Lake environment. Like Mountain Car, the Cart Pole environment's observation space is also continuous. However, it has a more complicated continuous observation space: the cart's position and velocity and the pole's angle and angular velocity.

##### Code Reference:
* [cartpole_q.py](https://github.com/johnnycode8/gym_solutions/blob/main/cartpole_q.py) 

##### YouTube Tutorial:
<a href='https://youtu.be/2u1REHeHMrg&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/2u1REHeHMrg/0.jpg' width='400' alt='Solves the CartPole-v1 with Q-Learning'/></a>


<br /><br />

# Deep Reinforcement Learning Tutorials

## Getting Started with Neural Networks
Before diving into Deep Reinforcement Learning, it would be helpful to have a basic understanding of Neural Networks. This hands-on end-to-end example of how to calculate Loss and Gradient Descent on the smallest network.

##### Code Reference:
* [Basic Neural Network](https://github.com/johnnycode8/basic_neural_network) repo

##### YouTube Tutorial:
<a href='https://youtu.be/6kOvmZDEMdc&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/6kOvmZDEMdc/0.jpg' width='400' alt='Work Thru the Most Basic Neural Network with Simplified Math and Python'/></a>

<br />

## Deep Q-Learning (DQL) Explained
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


##### Code Reference:
* [frozen_lake_dql.py](https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py)

##### Dependencies:
* <a href='https://pytorch.org/'>PyTorch</a>

##### YouTube Tutorial:
<a href='https://youtu.be/EUrWGTCGzlA&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/EUrWGTCGzlA/0.jpg' width='400' alt='Deep Q-Learning DQL/DQN Explained + Code Walkthru + Demo'/></a>

<br />

## Apply DQN to Gymnasium Mountain Car
We've already solve [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/) with Q-Learning (above). For learning purposes, we'll do it again with Deep Q-Learning. Hopefully, it'll give you a better understanding on how to adapt the code for your needs.

##### Code Reference:
* [mountain_car_dql.py](https://github.com/johnnycode8/gym_solutions/blob/main/mountain_car_dql.py)

##### Dependencies:
* <a href='https://pytorch.org/'>PyTorch</a>

##### YouTube Tutorial:
<a href='https://youtu.be/oceguqZxjn4&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/oceguqZxjn4/0.jpg' width='400' alt='Deep Q-Learning on MountainCar-V0'/></a>

<br />

## Get Started with Convolutional Neural Network (CNN)
In part 1 (above), the Deep Q-Networks (DQN) used were straightforward neural networks with a hidden layer and an output layer. This network architecture works for simple environments. However, for complex environments—such as Atari Pong—where the agent learns from the environment visually, we need to modify our DQNs with convolutional layers. We'll continue the explanation on the very simple [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 4x4 environment, however, we'll modify the inputs such that they are treated as images.

##### Code Reference:
* [frozen_lake_dql_cnn.py](https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql_cnn.py)

##### Dependencies:
* <a href='https://pytorch.org/'>PyTorch</a>

##### YouTube Tutorial:
<a href='https://youtu.be/qKePPepISiA&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/qKePPepISiA/0.jpg' width='400' alt='Deep Q-Learning with Convolutional Neural Networks'/></a>

<br /><br />

# Stable Baselines3 Tutorials
## Stable Baselines3: Get Started Guide | Train Gymnasium MuJoCo Humanoid-v4
Get started with the Stable Baselines3 Reinforcement Learning library by training the Gymnasium MuJoCo [Humanoid-v4](https://gymnasium.farama.org/environments/mujoco/humanoid/) environment with the Soft Actor-Critic (SAC) algorithm. The focus is on the usage of the Stable Baselines3 (SB3) library and the use of TensorBoard to monitor training progress. Other algorithms used in the demo include Twin Delayed Deep Deterministic Policy Gradient (TD3) and Advantage Actor Critic (A2C).

##### Code Reference:
* [sb3.py](https://github.com/johnnycode8/gym_solutions/blob/main/sb3.py) 

##### Dependency:
* [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/)

##### YouTube Tutorial:
<a href='https://youtu.be/OqvXHi_QtT0&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/OqvXHi_QtT0/0.jpg' width='400' alt='Train Gymnasium Humanoid-v4 with Stable Baselines3'/></a>

<br />

## Stable Baselines3 - Beginner's Guide to Choosing RL Algorithms for Training
SB3 offers many ready-to-use RL algorithms out of the box, but as beginners, how do we know which algorithms to use? We'll discuss this topic in the video:

<a href='https://youtu.be/2AFl-iWGQzc&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/2AFl-iWGQzc/0.jpg' width='400' alt='Beginners Guide on Choosing Stable Baselines3 Algorithms for Training'/></a>

<br />

## Stable Baselines3: Dynamically Load RL Algorithm for Training | Train Gymnasium Pendulum
In part 1, for simplicity, the algorithms (SAC, TD3, 2C) were hardcoded in the code. In part 2, we'll make loading and creating instances of the algorithms dynamic. To test the changes, we'll train [Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/) using SAC and TD3 simultaneously and monitor the progress thru TensorBoard.

##### Code Reference:
* [sb3v2.py](https://github.com/johnnycode8/gym_solutions/blob/main/sb3v2.py) 

##### YouTube Tutorial:
<a href='https://youtu.be/nf2IE2GEJ-s&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/nf2IE2GEJ-s/0.jpg' width='400' alt='Train Gymnasium Pendulum-v1 with Stable Baselines3'/></a>

<br />

## Stable Baselines3: Auto Stop Training When Best Model is Found | Train Gym Bipedal Walker
This tutorial walks thru the code that automatically stop training when the best model is found. We'll demonstrate by training the Gymnasium [BipedalWalker-v3](https://gymnasium.farama.org/environments/box2d/bipedal_walker/) using Soft-Actor Critic.

##### Code Reference:
* [sb3v3.py](https://github.com/johnnycode8/gym_solutions/blob/main/sb3v3.py) 

##### YouTube Tutorial:
<a href='https://youtu.be/mCkgLweyMqo&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte'><img src='https://img.youtube.com/vi/mCkgLweyMqo/0.jpg' width='400' alt='Train Gymnasium BipedalWalker-v3 with Stable Baselines3'/></a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
