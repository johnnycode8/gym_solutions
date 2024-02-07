<a name="readme-top"></a>

<h3 align="center">Gymnasium Reinforcement Learning Solutions</h3>

Collection of Python code that solves Reinforcement Learning environments from the [Gymnasium Library](https://gymnasium.farama.org/), formerly OpenAI’s Gym library. Each solution has a companion video explanation and code walkthrough from my YouTube channel [@johnnycode](https://www.youtube.com/@johnnycode). If the code and video helped you, please consider:  
<a href='https://www.buymeacoffee.com/johnnycode'><img src="https://cdn.buymeacoffee.com/buttons/default-blue.png" alt="Buy Me A Coffee" height="41" width="174"></a>

## Installation
The [Gymnasium Library](https://gymnasium.farama.org/) is supported on Linux and Mac OS, but not officially on Windows. On Windows, the Box2D package (Bipedal Walker, Car Racing, Lunar Lander) is problematic during installation, you may see errors such as:
* ERROR: Failed building wheels for box2d-py
* ERROR: Command swig.exe failed
* ERROR: Microsoft Visual C++ 14.0 or greater is required.

My Gymnasium on Windows installation video shows you how to resolve these errors and successfully install the complete set of Gymnasium Reinforcement Learning environments.

##### YouTube Tutorial:
<a href='https://youtu.be/gMgj4pSHLww'><img src='https://img.youtube.com/vi/gMgj4pSHLww/0.jpg' width='400' alt='Install Gymnasium on Windows'/></a>


## Deep Q-Learning
(Beginners should start with Q-Learning first, scroll down). This Deep Reinforcement Learning tutorial explains how the Deep Q-Learning (DQL) algorithm uses two neural networks: a Policy Deep Q-Network (DQN) and a Target DQN, to solve the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 4x4 environment. The Frozen Lake environment is very simple and straightforward, allowing us to focus on how DQL works. The Epsilon-Greedy algorithm and the Experience Replay technique are also used as part of DQL to help train the learning agent. The code referenced here is also walked through in the YouTube tutorial. PyTorch is used to build the DQNs.

##### Code Reference:
* [frozen_lake_dql.py](https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py)

##### Dependencies:
* <a href='https://pytorch.org/'>PyTorch</a>

##### YouTube Tutorial:
<a href='https://youtu.be/EUrWGTCGzlA'><img src='https://img.youtube.com/vi/EUrWGTCGzlA/0.jpg' width='400' alt='Deep Q-Learning DQL/DQN Explained + Code Walkthru + Demo'/></a>


## Q-Learning - Frozen Lake 8x8
This is the recommended starting point for beginners. This Q-Learning tutorial walks through the code on how to solve the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 8x8 map. The Frozen Lake environment is very simple and straightforward, allowing us to focus on how Q-Learning works. The Epsilon-Greedy algorithm is also used in conjunction with Q-Learning. Note that this tutorial does not explain the theory or math behind Q-Learning. 

##### Code Reference:
* [frozen_lake_q.py](https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_q.py) 

##### YouTube Tutorial:
<a href='https://youtu.be/ZhoIgo3qqLU'><img src='https://img.youtube.com/vi/ZhoIgo3qqLU/0.jpg' width='400' alt='Solve FrozenLake-v1 8x8 with Q-Learning'/></a>

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
<a href='https://youtu.be/1W_LOB-0IEY'><img src='https://img.youtube.com/vi/1W_LOB-0IEY/0.jpg' width='400' alt='See Q-Learning in Realtime on FrozenLake-v1'/></a>


## Q-Learning - Mountain Car
This Q-Learning tutorial solves the [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/) environment. It builds upon the code from the Frozen Lake environment. What is interesting about this environment is that the observation space is continuous, whereas the Frozen Lake environment's observation space is discrete. "Discrete" means that the agent, the elf in Frozen Lake, steps from one cell on the grid to the next, so there is a clear distinction that the agent is going from one state to another. "Continuous" means that the agent, the car in Mountain Car, traverses the mountain on a continuous road, with no clear distinction of states.

##### Code Reference:
* [mountain_car_q.py](https://github.com/johnnycode8/gym_solutions/blob/main/mountain_car_q.py) 

##### YouTube Tutorial:
<a href='https://youtu.be/_SWnNhM5w-g'><img src='https://img.youtube.com/vi/_SWnNhM5w-g/0.jpg' width='400' alt='Solves the MountainCar-v0 with Q-Learning'/></a>


## Q-Learning - Cart Pole
This Q-Learning tutorial solves the [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment. It builds upon the code from the Frozen Lake environment. Like Mountain Car, the Cart Pole environment's observation space is also continuous. However, it has a more complicated observation space, including the cart's position and velocity, as well as the pole's angle and angular velocity.

##### Code Reference:
* [cartpole_q.py](https://github.com/johnnycode8/gym_solutions/blob/main/cartpole_q.py) 

##### YouTube Tutorial:
<a href='https://youtu.be/2u1REHeHMrg'><img src='https://img.youtube.com/vi/2u1REHeHMrg/0.jpg' width='400' alt='Solves the CartPole-v1 with Q-Learning'/></a>


## StableBaseline 3
This Stable Baselines3 tutorial solves the [Humanoid-v4](https://gymnasium.farama.org/environments/mujoco/humanoid/) MuJoCo environment with the Soft Actor-Critic (SAC) algorithm. The focus is on the usage of the Stable Baselines3 library rather than the SAC algorithm. Other algorithms used in the demo include Twin Delayed Deep Deterministic Policy Gradient (TD3) and Advantage Actor Critic (A2C).

##### Code Reference:
* [sb3.py](https://github.com/johnnycode8/gym_solutions/blob/main/sb3.py) 

##### Dependency:
* [Stable Baseline3](https://stable-baselines3.readthedocs.io/en/master/)

##### YouTube Tutorial:
<a href='https://youtu.be/OqvXHi_QtT0'><img src='https://img.youtube.com/vi/OqvXHi_QtT0/0.jpg' width='400' alt='Solves the Humanoid-v4 with StableBaseline 3'/></a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
