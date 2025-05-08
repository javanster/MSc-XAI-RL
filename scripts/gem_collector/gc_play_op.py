import gem_collector
import gymnasium as gym
from tqdm import tqdm

if __name__ == "__main__":
    env = gym.make(id="GemCollector-v3", render_mode=None, show_raw_pixels=False)

    def get_optimal_action(step: int) -> int:
        """
        Optimal policy for GemCollector
        """
        return 1 - (step // 19) % 2

    log_vars = {
        "aquamarines_collected_count": 0,
        "amethysts_collected_count": 0,
        "emeralds_collected_count": 0,
        "rocks_collected_count": 0,
        "lava_termination_count": 0,
        "truncated_count": 0,
        "total_reward": 0,
    }

    EPISODES = 5000

    with tqdm(total=EPISODES, unit="episode") as pbar:
        for _ in range(EPISODES):
            step_n = 0
            terminated = False
            observation, _ = env.reset()
            while not terminated:
                env.render()
                action = get_optimal_action(step=step_n)

                observation, reward, terminated, truncated, info = env.step(action)

                step_n += 1

                log_vars["aquamarines_collected_count"] += info["aquamarine_collected"]
                log_vars["amethysts_collected_count"] += info["amethyst_collected"]
                log_vars["emeralds_collected_count"] += info["emerald_collected"]
                log_vars["rocks_collected_count"] += info["rocks_collected"]
                log_vars["lava_termination_count"] += 1 if info["lava_collision"] else 0
                log_vars["truncated_count"] += 1 if truncated else 0
                log_vars["total_reward"] += info["reward"]

            pbar.update(1)

    print(log_vars)
