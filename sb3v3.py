import gymnasium as gym
import stable_baselines3
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import argparse

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train():
    model = sb3_class('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    # Stop training when mean reward reaches reward_threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)

    # Stop training when model shows no improvement after max_no_improvement_evals, 
    # but do not start counting towards max_no_improvement_evals until after min_evals.
    # Number of timesteps before possibly stopping training = min_evals * eval_freq (below)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10000, verbose=1)

    eval_callback = EvalCallback(
        env, 
        eval_freq=10000, # how often to perform evaluation i.e. every 10000 timesteps.
        callback_on_new_best=callback_on_best, 
        callback_after_eval=stop_train_callback, 
        verbose=1, 
        best_model_save_path=os.path.join(model_dir, f"{args.gymenv}_{args.sb3_algo}"),
    )
    
    """
    total_timesteps: pass in a very large number to train (almost) indefinitely.
    tb_log_name: create log files with the name [gym env name]_[sb3 algorithm] i.e. Pendulum_v1_SAC
    callback: pass in reference to a callback fuction above
    """
    model.learn(total_timesteps=int(1e10), tb_log_name=f"{args.gymenv}_{args.sb3_algo}", callback=eval_callback)

def test():        
    model = sb3_class.load(os.path.join(model_dir, f"{args.gymenv}_{args.sb3_algo}", "best_model"), env=env)

    obs = env.reset()[0]   
    while True:
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. A2C, DDPG, DQN, PPO, SAC, TD3')    
    parser.add_argument('--test', help='Test mode', action='store_true')
    args = parser.parse_args()

    # Dynamic way to import algorithm. For example, passing in DQN is equivalent to hardcoding:
    # from stable_baselines3 import DQN
    sb3_class = getattr(stable_baselines3, args.sb3_algo)

    if args.test:
        env = gym.make(args.gymenv, render_mode='human')
        test()
    else:
        env = gym.make(args.gymenv)
        env = Monitor(env)
        # env = gym.wrappers.RecordVideo(env, video_folder=recording_dir, episode_trigger = lambda x: x % 10000 == 0)
        train()
        

