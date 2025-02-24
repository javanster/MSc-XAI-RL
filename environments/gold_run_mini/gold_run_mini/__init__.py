from gymnasium.envs.registration import register

register(
    id="GoldRunMini-v1",
    entry_point="gold_run_mini.envs:GoldRunMini",
    max_episode_steps=200,
)
