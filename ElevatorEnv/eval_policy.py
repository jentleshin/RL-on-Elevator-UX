import skvideo.io
from pyvirtualdisplay import Display
import gymnasium as gym

def save_video_of_model(env_name, model=None, suffix="", num_episodes=10):
    """
    Record a video that shows the behavior of an agent following a model 
    (i.e., policy) on the input environment
    """

    env = gym.make(env_name)
    obs= env.reset()
    prev_obs = obs

    filename = env_name + suffix + ".mp4"
    output_video = skvideo.io.FFmpegWriter(filename)

    counter = 0
    done = False
    num_runs = 0
    returns = 0
    while num_runs < num_episodes:
        frame = env.render()
        output_video.writeFrame(frame)

        if "Elevator" in env_name:
            input_obs = obs
        else:
            raise ValueError(f"Unknown env for saving: {env_name}")

        if model is not None:
            action = model(input_obs)
        else:
            action = env.action_space.sample()


        prev_obs = obs
        obs, reward, done, truncated, info = env.step(action)
        counter += 1
        returns += reward
        if done or truncated:
            num_runs += 1
            obs= env.reset()

    output_video.close()
    print("Successfully saved {} frames into {}!".format(counter, filename))
    return filename, returns / num_runs

def eval_policy(env_name,policy,episodes):
    _,rew=save_video_of_model(env_name,policy,num_episodes=episodes)
    print(rew)