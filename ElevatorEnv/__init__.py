from gymnasium.envs.registration import register

MAX_EPISODE_STEP=1000
REWARD_THRESHOLD=100

register(
    id='Elevator-v0',
    entry_point='ElevatorEnv.env:ElevatorEnv',
    max_episode_steps=MAX_EPISODE_STEP,
    reward_threshold=REWARD_THRESHOLD
)