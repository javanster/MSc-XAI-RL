from gymnasium.envs.registration import register

register(
    id="ChangingSupervisor-v1",
    entry_point="changing_supervisor.envs:ChangingSupervisor",
    max_episode_steps=400,
)
