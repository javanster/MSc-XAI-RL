from gymnasium.envs.registration import register

register(
    id="BoxEscape-v1",
    entry_point="box_escape.envs:BoxEscape",
    max_episode_steps=200,
)
