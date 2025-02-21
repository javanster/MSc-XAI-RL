from gymnasium.envs.registration import register

register(
    id="MinecartCounter-v2",
    entry_point="minecart_counter.envs:MinecartCounter",
    max_episode_steps=200,
)
