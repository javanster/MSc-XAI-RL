from gymnasium.envs.registration import register

register(
    id='FruitRun-v0',
    entry_point='fruit_run.envs:FruitRun',
    max_episode_steps=200,
)
