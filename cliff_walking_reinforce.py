"""
This is partial (modified) code from https://huggingface.co/learn/deep-rl-course/en/unit4/hands-on
"""

import gymnasium as gym
import numpy as np
from collections import deque

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        # Unsqueeze converts 1D=>2D [1,2,3,4]=>[[1,2,3,4]] for input into the NN
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state = state.unsqueeze(0).to(device)

        # Apply policy to state, return probabilities for each action
        probs = self.forward(state).cpu()

        # Randomly choose an action based on the probabilities
        m = Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)

def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    # Line 3 of pseudocode
    scores = []
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()[0]
        state = state_to_dqn_input(state, state_size)
        # state = np.array([state])
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _, _ = env.step(action)
            state = state_to_dqn_input(state, state_size)
            # state = np.array([state])
            rewards.append(float(reward))
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        """
        Compute the discounted returns at each timestep,
        as the sum of the gamma-discounted return at time t (G_t) + the reward at time t

        In O(N) time, where N is the number of time steps
        (this definition of the discounted return G_t follows the definition of this quantity
        shown at page 44 of Sutton&Barto 2017 2nd draft)
        G_t = r_(t+1) + r_(t+2) + ...

        Given this formulation, the returns at each timestep t can be computed
        by re-using the computed future returns G_(t+1) to compute the current return G_t
        G_t = r_(t+1) + gamma*G_(t+1)
        G_(t-1) = r_t + gamma* G_t
        (this follows a dynamic programming approach, with which we memorize solutions in order
        to avoid computing them multiple times)

        This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...


        Given the above, we calculate the returns at timestep t as:
            gamma[t] * return[t] + reward[t]

        We compute this starting from the last timestep to the first, in order
        to employ the formula presented above and avoid redundant computations that would be needed
        if we were to do it from first to last.

        Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        a normal python list would instead require O(N) to do this.
        """
        for t in range(n_steps)[::-1]:  # Loop in reverse order
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(gamma*disc_return_t + rewards[t])

        ## standardization of the returns (z-score) is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)   # Perform gradient descent on negative log_prob*disc_return is equivalent to performing gradient ascent.
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))    

    return scores


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()[0]
        state = state_to_dqn_input(state, state_size)
        # state = np.array([state])
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, _, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
            state = state_to_dqn_input(state, state_size)
            # state = np.array([state])
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

def state_to_dqn_input(state:int, num_states:int)->torch.Tensor:
    input_tensor = torch.zeros(num_states)
    input_tensor[state] = 1
    return input_tensor


if __name__ == '__main__':
    env_id = "CliffWalking-v0"
    # Create the env
    env = gym.make(env_id)

    # Create the evaluation env
    eval_env = gym.make(env_id, render_mode='human')

    # Get the state space and action space
    state_size = env.observation_space.n
    action_size = env.action_space.n

    hyperparameters = {
        "h_size": 16,   # nodes in hidden layer
        "n_training_episodes": 1000,
        "n_evaluation_episodes": 10,
        "max_t": 1000,  # max timesteps per episode
        "gamma": 1,     # discount factor
        "lr": 0.01,     # learning rate
        "env_id": env_id,
        "state_space": state_size,
        "action_space": action_size,
    }

    # Create policy and place it to the device
    policy = Policy(hyperparameters["state_space"], hyperparameters["action_space"], hyperparameters["h_size"]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=hyperparameters["lr"])
    scores = reinforce(policy,
                   optimizer,
                   hyperparameters["n_training_episodes"],
                   hyperparameters["max_t"],
                   hyperparameters["gamma"],
                   25)
    torch.save(policy.state_dict(), "cliff_walking_reinforce.pt")

    # Load learned policy
    policy.load_state_dict(torch.load("cliff_walking_reinforce.pt"))    
    mean_reward, std_reward = evaluate_agent(eval_env,
               hyperparameters["max_t"],
               hyperparameters["n_evaluation_episodes"],
               policy)
    print(f"eval mean reward {mean_reward}  std reward {std_reward}")


