from gymnasium import Env

from rl_tcav import BinaryConcept

from .constants import ENV_NAME


def _env_validation(env: Env) -> None:
    if env.unwrapped.name != ENV_NAME:
        raise ValueError("Incorrect env provided. Must be an instance of the GoldRunMini env")


def is_gold_above(env: Env) -> bool:
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


def is_lava_1_above(env: Env) -> bool:
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


def get_grm_concepts():
    c_gold_above = BinaryConcept(
        name="gold_above",
        observation_presence_callback=is_gold_above,
        environment_name=ENV_NAME,
    )

    c_lava_1_above = BinaryConcept(
        name="lava_1_above",
        observation_presence_callback=is_lava_1_above,
        environment_name=ENV_NAME,
    )

    return [c_gold_above, c_lava_1_above]
