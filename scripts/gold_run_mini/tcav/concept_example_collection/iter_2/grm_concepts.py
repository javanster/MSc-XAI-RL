import random

from gymnasium import Env

from rl_tcav import BinaryConcept

from .constants import ENV_NAME


def _env_validation(env: Env) -> None:
    if env.unwrapped.name != ENV_NAME:
        raise ValueError("Incorrect env provided. Must be an instance of the GoldRunMini env")


def is_gold_above(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    gold_chunks = env.unwrapped.gold_chunks
    for gc in gold_chunks:
        if gc.y < agent.y:
            return True
    return False


def is_gold_right(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    gold_chunks = env.unwrapped.gold_chunks
    for gc in gold_chunks:
        if gc.x > agent.x:
            return True
    return False


def is_gold_down(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    gold_chunks = env.unwrapped.gold_chunks
    for gc in gold_chunks:
        if gc.y > agent.y:
            return True
    return False


def is_gold_left(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    gold_chunks = env.unwrapped.gold_chunks
    for gc in gold_chunks:
        if gc.x < agent.x:
            return True
    return False


def is_lava_1_above(env: Env) -> bool:
    """
    Checks if a spot of lava is exactly 1 cell above the agent, x coordinates being equal, in the
    current state of the given instance of the GoldRunMini environment.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    lava_spots = env.unwrapped.lava
    for ls in lava_spots:
        if ls.x == agent.x and ls.y == agent.y - 1:
            return True
    return False


def is_lava_1_right(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    lava_spots = env.unwrapped.lava
    for ls in lava_spots:
        if ls.x - 1 == agent.x and ls.y == agent.y:
            return True
    return False


def is_lava_1_below(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    lava_spots = env.unwrapped.lava
    for ls in lava_spots:
        if ls.x == agent.x and ls.y == agent.y + 1:
            return True
    return False


def is_lava_1_left(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    lava_spots = env.unwrapped.lava
    for ls in lava_spots:
        if ls.x + 1 == agent.x and ls.y == agent.y:
            return True
    return False


def is_green_exit_above(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    green_exit = env.unwrapped.passage
    if green_exit is None:
        return False
    if green_exit.y < agent.y:
        return True
    return False


def is_green_exit_right(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    green_exit = env.unwrapped.passage
    if green_exit is None:
        return False
    if green_exit.x > agent.x:
        return True
    return False


def is_green_exit_down(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    green_exit = env.unwrapped.passage
    if green_exit is None:
        return False
    if green_exit.y > agent.y:
        return True
    return False


def is_green_exit_left(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    green_exit = env.unwrapped.passage
    if green_exit is None:
        return False
    if green_exit.x < agent.x:
        return True
    return False


def is_purple_exit_above(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    purple_exit = env.unwrapped.early_term_passage
    if purple_exit is None:
        return False
    if purple_exit.y < agent.y:
        return True
    return False


def is_purple_exit_right(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    purple_exit = env.unwrapped.early_term_passage
    if purple_exit is None:
        return False
    if purple_exit.x > agent.x:
        return True
    return False


def is_purple_exit_down(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    purple_exit = env.unwrapped.early_term_passage
    if purple_exit is None:
        return False
    if purple_exit.y > agent.y:
        return True
    return False


def is_purple_exit_left(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    purple_exit = env.unwrapped.early_term_passage
    if purple_exit is None:
        return False
    if purple_exit.x < agent.x:
        return True
    return False


def is_wall_directly_above(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    return agent.y == 1


def is_wall_directly_right(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    return agent.x == 9


def is_wall_directly_below(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    return agent.y == 9


def is_wall_directly_left(env: Env) -> bool:
    _env_validation(env)
    agent = env.unwrapped.agent
    return agent.x == 1


def random_binary(env: Env) -> bool:
    _env_validation(env)
    return random.random() > 0.5


def get_grm_concepts():
    return [
        # Gold position concepts
        BinaryConcept(
            name="gold_above",
            observation_presence_callback=is_gold_above,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="gold_right",
            observation_presence_callback=is_gold_right,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="gold_down", observation_presence_callback=is_gold_down, environment_name=ENV_NAME
        ),
        BinaryConcept(
            name="gold_left", observation_presence_callback=is_gold_left, environment_name=ENV_NAME
        ),
        # Lava proximity concepts
        BinaryConcept(
            name="lava_1_above",
            observation_presence_callback=is_lava_1_above,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="lava_1_right",
            observation_presence_callback=is_lava_1_right,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="lava_1_below",
            observation_presence_callback=is_lava_1_below,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="lava_1_left",
            observation_presence_callback=is_lava_1_left,
            environment_name=ENV_NAME,
        ),
        # Green exit (final passage) location
        BinaryConcept(
            name="green_exit_above",
            observation_presence_callback=is_green_exit_above,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="green_exit_right",
            observation_presence_callback=is_green_exit_right,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="green_exit_down",
            observation_presence_callback=is_green_exit_down,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="green_exit_left",
            observation_presence_callback=is_green_exit_left,
            environment_name=ENV_NAME,
        ),
        # Purple exit (early termination) location
        BinaryConcept(
            name="purple_exit_above",
            observation_presence_callback=is_purple_exit_above,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="purple_exit_right",
            observation_presence_callback=is_purple_exit_right,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="purple_exit_down",
            observation_presence_callback=is_purple_exit_down,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="purple_exit_left",
            observation_presence_callback=is_purple_exit_left,
            environment_name=ENV_NAME,
        ),
        # Wall boundary concepts
        BinaryConcept(
            name="wall_directly_above",
            observation_presence_callback=is_wall_directly_above,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="wall_directly_right",
            observation_presence_callback=is_wall_directly_right,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="wall_directly_below",
            observation_presence_callback=is_wall_directly_below,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="wall_directly_left",
            observation_presence_callback=is_wall_directly_left,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="random_binary",
            observation_presence_callback=random_binary,
            environment_name=ENV_NAME,
        ),
    ]
