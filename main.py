import argparse

import gymnasium as gym
import skvideo.io
import ElevatorEnv
import ElevatorEnv.baseline_policy as baseline_policy

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import matplotlib.pyplot as plt

def main(args):
    
    if args.mode == "train":
        vec_env = make_vec_env('Elevator-v0', n_envs=8)
        #vec_env = make_vec_env('Elevator-v1', n_envs=8)
        if args.load == "None":
            model = PPO('MultiInputPolicy', vec_env, verbose=1)
        else:
            model = PPO.load(f"./checkpoint/{args.load}")
            model.set_env(vec_env)
        model.learn(total_timesteps=args.timesteps)
        model.save(f"./checkpoint/{args.checkpoint}")
        print("Successfully saved trained model!")
        return

    elif args.mode == "test":
        # env=gym.make('Elevator-v0')
        # env.reset()
        # plt.imshow(env.render())
        # plt.show()


        model = PPO.load(f"./checkpoint/{args.checkpoint}")
        vec_env = make_vec_env('Elevator-v0', n_envs=1)
        #vec_env = make_vec_env('Elevator-v1', n_envs=1)
        obs = vec_env.reset()

        output_video = skvideo.io.FFmpegWriter(f"./video/{args.filename}.mp4")
        counter=0
        num_runs=0
        reward=0
        while num_runs < args.num_episodes:
            action, _ = model.predict(obs)
            obs, rew, done, metric = vec_env.step(action)
            frame = vec_env.render()
            output_video.writeFrame(frame)
            counter += 1
            reward+=rew
            if done:
                print("{}th run".format(num_runs))
                print("Reward: {}".format(reward))
                tot_passengers,tot_waiting_time,avg_delayed_time,visited_floors,rms_avg_actions,_,_,_=metric[0].values()
                print("tot_passengers: {}".format(tot_passengers))
                print("tot_waiting_time: {}".format(tot_waiting_time))
                print("avg_delayed_time: {}".format(avg_delayed_time))
                print("visited_floors: {}".format(visited_floors))
                print("rms_avg_actions: {}".format(rms_avg_actions))
                obs = vec_env.reset()
                reward=0
                num_runs += 1

        print("Successfully saved {} frames into {}!".format(counter, args.filename))

    elif args.mode == "baseline":
        env=gym.make('Elevator-v0')
        state,_=env.reset()
        policy=baseline_policy.baseline_policy()
        output_video = skvideo.io.FFmpegWriter(f"./video/{args.filename}.mp4")
        counter=0
        num_runs=0
        reward=0
        while num_runs < args.num_episodes:            
            action=policy.action(state)
            state,rew,done,_,_=env.step(action)            
            frame = env.render()
            output_video.writeFrame(frame)
            counter += 1
            reward+=rew
            if counter>1000:
                print("{}th run".format(num_runs))
                print("Reward: {}".format(reward))
                env.print_metric()
                state,_ = env.reset()
                num_runs += 1
                
                reward=0
                counter=0
                policy=baseline_policy.baseline_policy()
        print("Successfully saved {} frames into {}!".format(counter, args.filename))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode")

    # Subparser for the "train" mode
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--timesteps", type=int, default=10000000)
    train_parser.add_argument("--load", type=str, default="None")
    train_parser.add_argument("--checkpoint", type=str, default="recent")

    # Subparser for the "test" mode
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--num_episodes", type=int, default=10)
    test_parser.add_argument("--checkpoint", type=str, default="recent")
    test_parser.add_argument("--filename", type=str, default="recent")

    test_parser = subparsers.add_parser("baseline")
    test_parser.add_argument("--num_episodes", type=int, default=10)
    test_parser.add_argument("--filename", type=str, default="baseline")


    args = parser.parse_args()
    main(args)