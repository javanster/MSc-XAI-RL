from gymnasium.envs.registration import register

register(
    id="BoxEscapeLite-v1",
    entry_point="box_escape_lite.envs:BoxEscapeLite",
    max_episode_steps=200,
)
