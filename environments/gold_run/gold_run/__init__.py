from gymnasium.envs.registration import register

register(
    id="GoldRun-v2",
    entry_point="gold_run.envs:GoldRun",
    max_episode_steps=200,
)
