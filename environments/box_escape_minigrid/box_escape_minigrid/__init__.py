from gymnasium.envs.registration import register

register(
    id="BoxEscapeMiniGrid-v1",
    entry_point="box_escape_minigrid.envs:BoxEscapeMiniGrid",
    max_episode_steps=200,
)
