from gymnasium import Env

from rl_tcav import BinaryConcept

from .constants import ENV_NAME


def _env_validation(env: Env) -> None:
    if env.unwrapped.name != ENV_NAME:
        raise ValueError("Incorrect env provided. Must be an instance of the GemCollector env")


def is_aquamarine_left(env: Env) -> bool:
    """
    Checks whether an aquamarine is left of the agent, no matter the y-coordinate, in the current
    state of the given GemCollector environment instance.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    aquamarines = env.unwrapped.obj_lists["aquamarine"]
    for aquamarine in aquamarines:
        if aquamarine.x < agent.x:
            return True
    return False


def is_lava_1_above(env: Env) -> bool:
    """
    Checks whether a cell of lava is exactly 1 step above the agent, x-coordinates being equal, in
    the current state of the given GemCollector environment instance.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    lava = env.unwrapped.obj_lists["lava"]
    for lava_cell in lava:
        if lava_cell.x == agent.x and lava_cell.y == agent.y - 1:
            return True
    return False


def get_gc_concepts():
    c_aquamarine_left = BinaryConcept(
        name="aquamarine_left",
        observation_presence_callback=is_aquamarine_left,
        environment_name=ENV_NAME,
    )

    c_lava_1_above = BinaryConcept(
        name="lava_1_above",
        observation_presence_callback=is_lava_1_above,
        environment_name=ENV_NAME,
    )

    return [c_aquamarine_left, c_lava_1_above]
