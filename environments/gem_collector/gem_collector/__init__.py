from gymnasium.envs.registration import register

register(
    id="GemCollector-v2",
    entry_point="gem_collector.envs:GemCollector",
    max_episode_steps=190,
)
