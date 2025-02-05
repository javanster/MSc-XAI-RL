import gem_collector
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make(id="GemCollector-v3", render_mode="human", show_raw_pixels=False)

    def get_optimal_action(step: int) -> int:
        """
        Optimal policy for GemCollector
        """
        return 1 - (step // 19) % 2

    for _ in range(1):
        step_n = 0
        terminated = False
        observation, _ = env.reset()
        while not terminated:
            env.render()
            action = get_optimal_action(step=step_n)

            observation, reward, terminated, truncated, _ = env.step(action)

            step_n += 1
