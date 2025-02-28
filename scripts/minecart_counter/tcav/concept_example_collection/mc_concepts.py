from gymnasium import Env

from rl_tcav import BinaryConcept, ContinuousConcept

from .constants import ENV_NAME


def _env_validation(env: Env) -> None:
    if env.unwrapped.name != ENV_NAME:
        raise ValueError("Incorrect env provided. Must be an instance of the MinecartCounter env")


def minecarts_n(env: Env) -> float:
    """
    Counts the number of minecarts present in the current state of the given instance of the
    MinecartCounter environment.
    """
    _env_validation(env)
    return float(len(env.unwrapped.minecarts))


def minecart_1_left(env: Env) -> bool:
    """
    Checks if a minecart is exactly 1 cell left of the agent, x-coordinates being equal, in the
    current state of the given instance of the MinecartCounter environment.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    minecarts = env.unwrapped.minecarts
    for minecart in minecarts:
        if minecart.y == agent.y and minecart.x == agent.x - 1:
            return True
    return False


def get_mc_continuous_concepts():
    c_minecarts_n = ContinuousConcept(
        name="minecarts_n",
        observation_presence_callback=minecarts_n,
        environment_name=ENV_NAME,
    )

    return [c_minecarts_n]


def get_mc_binary_concepts():
    c_minecart_1_left = BinaryConcept(
        name="minecart_1_left",
        observation_presence_callback=minecart_1_left,
        environment_name=ENV_NAME,
    )

    return [c_minecart_1_left]
