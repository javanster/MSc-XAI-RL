from gymnasium.envs.registration import register

register(
    id="CCR-v5",
    entry_point="ccr.envs:CarRacing",
)
