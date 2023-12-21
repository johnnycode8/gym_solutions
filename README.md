<a name="readme-top"></a>

<h3 align="center">Gymnasium Reinforcement Learning Solutions</h3>

Collection of Python code that solves Reinforcement Learning environments from the [Gymnasium Library](https://gymnasium.farama.org/), formerly OpenAI’s Gym library. Each solution has a companion video explanation and code walkthrough from my YouTube channel [@johnnycode](https://www.youtube.com/@johnnycode). If the code and video helped you, please consider:  
<a href='https://www.buymeacoffee.com/johnnycode'><img src="https://cdn.buymeacoffee.com/buttons/default-blue.png" alt="Buy Me A Coffee" height="41" width="174"></a>

## Installation
The [Gymnasium Library](https://gymnasium.farama.org/) is supported on Linux and Mac OS, but not officially on Windows. On Windows, the Box2D package (Bipedal Walker, Car Racing, Lunar Lander) is problematic during installation, you may see errors such as:
* ERROR: Failed building wheels for box2d-py
* ERROR: Command swig.exe failed
* ERROR: Microsoft Visual C++ 14.0 or greater is required.

I have been developing successfully on Windows. If you encounter these errors, check out my Gymnasium on Windows installation guide.

##### Video Tutorial:
<a href='https://youtu.be/gMgj4pSHLww'><img src='https://img.youtube.com/vi/gMgj4pSHLww/0.jpg' width='400' alt='Install Gymnasium on Windows'/></a>


## Deep Q-Learning
This Deep Reinforcement Learning tutorial explains how the Deep Q-Learning (DQL) algorithm uses a Policy Deep Q-Network (DQN) and a Target DQN to solve the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 4x4 environment. Beginners should start with Q-Learning first (scroll down).

##### Code Reference:
* frozen_lake_dql.py

##### Dependencies:
* <a href='https://pytorch.org/'>PyTorch</a>

##### Video Tutorial:
<a href='https://youtu.be/EUrWGTCGzlA'><img src='https://img.youtube.com/vi/EUrWGTCGzlA/0.jpg' width='400' alt='Deep Q-Learning DQL/DQN Explained + Code Walkthru + Demo'/></a>


## Q-Learning - Frozen Lake 8x8
This Q-Learning tutorial solves the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 8x8 map. This is the recommended starting point for beginners.

##### Code Reference:
* frozen_lake_q.py

##### Video Tutorial:
<a href='https://youtu.be/ZhoIgo3qqLU'><img src='https://img.youtube.com/vi/ZhoIgo3qqLU/0.jpg' width='400' alt='Solve FrozenLake-v1 8x8 with Q-Learning'/></a>

## Q-Learning - Frozen Lake 8x8 Enhanced
This is the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment "enhanced" to help you better understand Q-Learning. Features:
* The Q values are overlayed on top of each cell of the map, so that you can visually see the Q values update in realtime while training!
* The map is enlarged to fill the whole screen so that it is easier to read the overlayed Q values.
* Shortcut keys to speed up or slow down the animation.

##### Code Reference:
* frozen_lake_qe.py 
This file is almost identical to frozen_lake_q.py above, except this uses the frozen_lake_enhanced.py environment. 
* frozen_lake_enhanced.py 
This is the FrozenLake-v1 environment overlayed with Q values. You do not need to understand this code.

##### Video Tutorial:
<a href='https://youtu.be/1W_LOB-0IEY'><img src='https://img.youtube.com/vi/1W_LOB-0IEY/0.jpg' width='400' alt='See Q-Learning in Realtime on FrozenLake-v1'/></a>


## Q-Learning - Mountain Car
This Q-Learning tutorial solves the [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/) environment. What is interesting about this environment is that the observation space is continuous. , whereas the Frozen Lake environment's observation space is discrete.

##### Code Reference:
* mountain_car_q.py

##### Video Tutorial:
<a href='https://youtu.be/_SWnNhM5w-g'><img src='https://img.youtube.com/vi/_SWnNhM5w-g/0.jpg' width='400' alt='Solves the MountainCar-v0 with Q-Learning'/></a>


## Q-Learning - Cart Pole
This Q-Learning tutorial solves the [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment. Like Mountain Car, the Cart Pole environment's observation space is also continuous. However, it is a more complicated observation space: cart's position and velocity and the pole's angle and angular velocity.

##### Code Reference:
* cartpole_q.py

##### Video Tutorial:
<a href='https://youtu.be/2u1REHeHMrg'><img src='https://img.youtube.com/vi/2u1REHeHMrg/0.jpg' width='400' alt='Solves the CartPole-v1 with Q-Learning'/></a>


## StableBaseline 3
This Stable Baselines3 tutorial solves the [Humanoid-v4](https://gymnasium.farama.org/environments/mujoco/humanoid/) environment with the Soft Actor-Critic algorithm. The focus is on the Stable Baselines3 library rather than the SAC algorithm.

##### Code Reference:
* sb3.py

##### Dependency:
* [Stable Baseline3](https://stable-baselines3.readthedocs.io/en/master/)

##### Video Tutorial:
<a href='https://youtu.be/OqvXHi_QtT0'><img src='https://img.youtube.com/vi/OqvXHi_QtT0/0.jpg' width='400' alt='Solves the Humanoid-v4 with StableBaseline 3'/></a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
