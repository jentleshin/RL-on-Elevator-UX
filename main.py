import argparse
import sys
#import torch
import gymnasium as gym
import ElevatorEnv
from ElevatorEnv.eval_policy import eval_policy
import matplotlib.pyplot as plt
import numpy as np

from PPO.ppo import PPO
from PPO.network import FeedForwardNN
#from PPO.eval_policy import eval_policy


# def train(env, hyperparameters, actor_model, critic_model):
# 	"""
# 		Trains the model.

# 		Parameters:
# 			env - the environment to train on
# 			hyperparameters - a dict of hyperparameters to use, defined in main
# 			actor_model - the actor model to load in if we want to continue training
# 			critic_model - the critic model to load in if we want to continue training

# 		Return:
# 			None
# 	"""	
# 	print(f"Training", flush=True)
# 	# Create a model for PPO.
# 	model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

# 	# Tries to load in an existing actor/critic model to continue training on
# 	if actor_model != '' and critic_model != '':
# 		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
# 		model.actor.load_state_dict(torch.load(actor_model))
# 		model.critic.load_state_dict(torch.load(critic_model))
# 		print(f"Successfully loaded.", flush=True)
# 	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
# 		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
# 		sys.exit(0)
# 	else:
# 		print(f"Training from scratch.", flush=True)

# 	# Train the PPO model with a specified total timesteps
# 	# NOTE: You can change the total timesteps here, I put a big number just because
# 	# you can kill the process whenever you feel like PPO is converging
# 	model.learn(total_timesteps=200_000_000)

# def test(env, actor_model):
# 	"""
# 		Tests the model.

# 		Parameters:
# 			env - the environment to test the policy on
# 			actor_model - the actor model to load in

# 		Return:
# 			None
# 	"""
# 	print(f"Testing {actor_model}", flush=True)

# 	# If the actor model is not specified, then exit
# 	if actor_model == '':
# 		print(f"Didn't specify model file. Exiting.", flush=True)
# 		sys.exit(0)

# 	# Extract out dimensions of observation and action spaces
# 	obs_dim = env.observation_space.shape[0]
# 	act_dim = env.action_space.shape[0]

# 	# Build our policy the same way we build our actor model in PPO
# 	policy = FeedForwardNN(obs_dim, act_dim)

# 	# Load in the actor model saved by the PPO algorithm
# 	policy.load_state_dict(torch.load(actor_model))

# 	# Evaluate our policy with a separate module, eval_policy, to demonstrate
# 	# that once we are done training the model/policy with ppo.py, we no longer need
# 	# ppo.py since it only contains the training algorithm. The model/policy itself exists
# 	# independently as a binary file that can be loaded in with torch.
# 	eval_policy(policy=policy, env=env, render=True)


def dummy_policy(state):
    return np.random.rand(1).item()*20.0 -10.0


def main(args):
	"""
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
	# NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
	# ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
	# To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
	hyperparameters = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': True,
				'render_every_i': 10
			  }

	# Creates the environment we'll be running. If you want to replace with your own
	# custom environment, note that it must inherit Gym and have both continuous
	# observation and action spaces.

	env = gym.make('Elevator-v0')
	env.reset()
	# obs_dim = env.observation_space.shape
	# act_dim = env.action_space.shape
	# print(obs_dim)
	# print(act_dim)
	# plt.imshow(env.render())
	# plt.show()
	eval_policy('Elevator-v0',policy=dummy_policy,episodes=10)
	
    
	
	# Train or test, depending on the mode specified
	# if args.mode == 'train':
	# 	train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
	# else:
	# 	test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # your actor model filename
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   # your critic model filename

	args = parser.parse_args() # Parse arguments from command line
	
	main(args)