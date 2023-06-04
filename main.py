import argparse

import gymnasium as gym
import skvideo.io
import ElevatorEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def main(args):
    
    if args.mode == "train":
        vec_env = make_vec_env('Elevator-v0', n_envs=8)
        model = PPO('MultiInputPolicy', vec_env, verbose=1)
        model.learn(total_timesteps=args.timesteps)
        model.save(f"./checkpoint/{args.checkpoint}")
        print("Successfully saved trained model!")
        return

    elif args.mode == "test":

        model = PPO.load(f"./checkpoint/{args.checkpoint}")
        vec_env = make_vec_env('Elevator-v0', n_envs=1)
        obs = vec_env.reset()

        output_video = skvideo.io.FFmpegWriter(f"./video/{args.filename}.mp4")
        counter=0
        num_runs=0
        while num_runs < args.num_episodes:
            action, _ = model.predict(obs)
            obs, _, done, _ = vec_env.step(action)
            frame = vec_env.render()
            output_video.writeFrame(frame)
            counter += 1

            if done:
                obs = vec_env.reset()
                num_runs += 1

        print("Successfully saved {} frames into {}!".format(counter, args.filename))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode")

    # Subparser for the "train" mode
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--timesteps", type=int, default=250000)
    train_parser.add_argument("--checkpoint", type=str, default="recent")

    # Subparser for the "test" mode
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--num_episodes", type=int, default=10)
    test_parser.add_argument("--checkpoint", type=str, default="recent")
    test_parser.add_argument("--filename", type=str, default="recent")

    args = parser.parse_args()
    main(args)