from gymnasium import Env


def _env_validation(env: Env) -> None:
    if env.unwrapped.name != "gem_collector":
        raise ValueError("Incorrect env provided. Must be an instance of the GemCollector env")


def aquamarine_left(env: Env) -> bool:
    """
    Checks whether an aquamarine is left of the agent, no matter the y-coordinate, in the current
    state of the given GemCollector environment instance.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    aquamarines = env.obj_lists["aquamarine"]
    for aquamarine in aquamarines:
        if aquamarine.x < agent.x:
            return True
    return False


def lava_1_above(env: Env) -> bool:
    """
    Checks whether a cell of lava is exactly 1 step above the agent, x-coordinates being equal, in
    the current state of the given GemCollector environment instance.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    lava = env.obj_lists["lava"]
    for lava_cell in lava:
        if lava_cell.x == agent.x and lava_cell.y == agent.y - 1:
            return True
    return False
