import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(is_training=True, render=False):

    env = gym.make('MountainCarContinuous-v0', render_mode='human' if render else None)

    # hyperparameters
    learning_rate_a = 0.9999        # alpha aka learning rate
    discount_factor_g = 0.9      # gamma aka discount factor.
    epsilon = 1                  # start episilon at 1 (100% random actions)
    epsilon_decay_rate = 0.001   # epsilon decay rate
    epsilon_min = 0.05           # minimum epsilon
    pos_divisions = 20           # used to convert continuous state space to discrete space
    vel_divisions = 20           # used to convert continuous state space to discrete space
    act_divisions = 10           # used to convert continuous action space to discrete space


    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], pos_divisions)    # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], vel_divisions)    # Between -0.07 and 0.07

    # Divide action space into discrete segments
    act_space = np.linspace(env.action_space.low[0], env.action_space.high[0], act_divisions, endpoint=False)  # Between -1.0 and 1.0
    act_lookup_space =np.append(act_space, 1)

    if(is_training):
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(act_space)+1)) # init a 21x21x11 array
    else:
        # Load q table
        f = open('mountain_car_cont.pkl', 'rb')
        q = pickle.load(f)
        f.close()


    best_reward = -999999        # track best reward
    best_mean_reward = -999999   # track best mean reward
    rewards_per_episode = []     # list to store rewards for each episode
    epsilon_history = []         # List to keep track of epsilon decay
    i = 0                        # episode counter

    while(True):

        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False          # True when reached goal
        rewards=0
        steps=0

        # Run episode until terminated or number of steps taken > 5000 (5000 is enough for the car to get to the goal randomly)
        while(not terminated and steps<5000):

            if is_training and np.random.rand() < epsilon:
                # Choose random action
                action = env.action_space.sample()

                # Discretize action space
                action_idx = np.digitize(action, act_space)
            else:
                # Choose action with highest Q value
                action_idx = np.argmax(q[state_p, state_v, :])

                # Convert discrete action back to continuous
                action = act_lookup_space[action_idx]

            # Execute action
            new_state,reward,terminated,_,_ = env.step(np.array([action]))

            # Discretize new state space
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                # Update Q table
                q[state_p, state_v, action_idx] = q[state_p, state_v, action_idx] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v,:]) - q[state_p, state_v, action_idx]
                )

            # Update state
            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            # Collect rewards
            rewards+=reward

            steps+=1

        # Track best reward
        if rewards > best_reward:
            best_reward = rewards

        # Store rewards per episode
        rewards_per_episode.append(rewards)

        # Print stats
        if is_training and i!=0 and i%100==0:
            # Calculate mean reward over the last 100 episodes
            mean_reward = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])
            print(f'Episode: {i}, Epsilon: {epsilon:0.2f}, Last Reward: {rewards:0.1f}, Best Reward: {best_reward:0.1f}, Mean Rewards {mean_reward:0.1f}')

            # Graph mean rewards
            mean_rewards = []
            for t in range(i):
                # Calculate mean reward over the t-100 episodes
                mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))

            plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
            plt.plot(mean_rewards)
            plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
            plt.plot(epsilon_history)
            plt.savefig(f'mountain_car_cont.png')

            # Save Q table to file on new best reward
            if mean_reward>best_mean_reward:
                best_mean_reward = mean_reward
                print(f'Saving model on new best mean reward: {best_mean_reward:0.1f}')
                f = open('mountain_car_cont.pkl','wb')
                pickle.dump(q, f)
                f.close()

            # Stop, if solved
            # Lower the reward threshold since the mean reward might never get above the threshold
            if mean_reward>env.spec.reward_threshold:
                break

        elif not is_training:
            print(f'Episode: {i} Rewards: {rewards:0.1f}')



        # Decay epsilon
        epsilon = max(epsilon - epsilon_decay_rate, epsilon_min)
        epsilon_history.append(epsilon)

        # Increment episode counter
        i+=1

    env.close()


if __name__ == '__main__':
    # Optional: Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Test model')
    args = parser.parse_args()

    if args.test:
        run(is_training=False, render=True)
    else:
        run(is_training=True, render=False)
