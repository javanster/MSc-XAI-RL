from gymnasium import Env


def _env_validation(env: Env) -> None:
    if env.unwrapped.name != "gold_run_mini":
        raise ValueError("Incorrect env provided. Must be an instance of the GoldRunMini env")


def gold_above(env: Env) -> bool:
    """
    Checks if a gold chunk is at least 1 cell above the agent, no matter the x coordinate, in the
    current state of the given instance of the GoldRunMini environment.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    gold_chunks = env.unwrapped.gold_chunks
    for gc in gold_chunks:
        if gc.y < agent.y:
            return True
    return False


def lava_1_above(env: Env) -> bool:
    """
    Checks if a spot of lava is exaclty 1 cell above the agent, x coordinates being equal, in the
    current state of the given instance of the GoldRunMini environment.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    lava_spots = env.unwrapped.lava
    for ls in lava_spots:
        if ls.x == agent.x and ls.y == agent.y - 1:
            return True
    return False
