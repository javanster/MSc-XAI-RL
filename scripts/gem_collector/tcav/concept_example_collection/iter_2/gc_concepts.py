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


def is_aquamarine_right(env: Env) -> bool:
    """
    Checks whether an aquamarine is right of the agent, no matter the y-coordinate, in the current
    state of the given GemCollector environment instance.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    aquamarines = env.unwrapped.obj_lists["aquamarine"]
    for aquamarine in aquamarines:
        if aquamarine.x > agent.x:
            return True
    return False


def is_aquamarine_above(env: Env) -> bool:
    """
    Checks whether an aquamarine is above the agent, i.e. on the same x-coordinate but with a smaller y-coordinate,
    in the current state of the given GemCollector environment instance.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    aquamarines = env.unwrapped.obj_lists["aquamarine"]
    for aquamarine in aquamarines:
        if aquamarine.x == agent.x and aquamarine.y < agent.y:
            return True
    return False


def is_emerald_left(env: Env) -> bool:
    """
    Checks whether an emerald is left of the agent, no matter the y-coordinate, in the current
    state of the given GemCollector environment instance.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    emeralds = env.unwrapped.obj_lists["emerald"]
    for emerald in emeralds:
        if emerald.x < agent.x:
            return True
    return False


def is_emerald_right(env: Env) -> bool:
    """
    Checks whether an emerald is right of the agent, no matter the y-coordinate, in the current
    state of the given GemCollector environment instance.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    emeralds = env.unwrapped.obj_lists["emerald"]
    for emerald in emeralds:
        if emerald.x > agent.x:
            return True
    return False


def is_emerald_above(env: Env) -> bool:
    """
    Checks whether an emerald is above the agent, i.e. on the same x-coordinate but with a smaller y-coordinate,
    in the current state of the given GemCollector environment instance.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    emeralds = env.unwrapped.obj_lists["emerald"]
    for emerald in emeralds:
        if emerald.x == agent.x and emerald.y < agent.y:
            return True
    return False


def is_amethyst_left(env: Env) -> bool:
    """
    Checks whether an amethyst is left of the agent, no matter the y-coordinate, in the current
    state of the given GemCollector environment instance.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    amethysts = env.unwrapped.obj_lists["amethyst"]
    for amethyst in amethysts:
        if amethyst.x < agent.x:
            return True
    return False


def is_amethyst_right(env: Env) -> bool:
    """
    Checks whether an amethyst is right of the agent, no matter the y-coordinate, in the current
    state of the given GemCollector environment instance.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    amethysts = env.unwrapped.obj_lists["amethyst"]
    for amethyst in amethysts:
        if amethyst.x > agent.x:
            return True
    return False


def is_amethyst_above(env: Env) -> bool:
    """
    Checks whether an amethyst is above the agent, i.e. on the same x-coordinate but with a smaller y-coordinate,
    in the current state of the given GemCollector environment instance.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    amethysts = env.unwrapped.obj_lists["amethyst"]
    for amethyst in amethysts:
        if amethyst.x == agent.x and amethyst.y < agent.y:
            return True
    return False


def is_aquamarine_left_within_reach(env: Env) -> bool:
    """
    Checks whether an aquamarine is to the left of the agent and within reach, considering
    that the aquamarine moves down one cell per step and the agent can move one cell
    horizontally per step.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    aquamarines = env.unwrapped.obj_lists["aquamarine"]
    for aquamarine in aquamarines:
        if aquamarine.x < agent.x and (agent.x - aquamarine.x <= agent.y - aquamarine.y):
            return True
    return False


def is_amethyst_left_within_reach(env: Env) -> bool:
    """
    Checks whether an amethyst is to the left of the agent and within reach, considering
    that the amethyst moves down one cell per step and the agent can move one cell
    horizontally per step.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    amethysts = env.unwrapped.obj_lists["amethyst"]
    for amethyst in amethysts:
        if amethyst.x < agent.x and (agent.x - amethyst.x <= agent.y - amethyst.y):
            return True
    return False


def is_emerald_left_within_reach(env: Env) -> bool:
    """
    Checks whether an emerald is to the left of the agent and within reach, considering
    that the emerald moves down one cell per step and the agent can move one cell
    horizontally per step.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    emeralds = env.unwrapped.obj_lists["emerald"]
    for emerald in emeralds:
        if emerald.x < agent.x and (agent.x - emerald.x <= agent.y - emerald.y):
            return True
    return False


def is_aquamarine_right_within_reach(env: Env) -> bool:
    """
    Checks whether an aquamarine is to the right of the agent and within reach, considering
    that the aquamarine moves down one cell per step and the agent can move one cell
    horizontally per step.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    aquamarines = env.unwrapped.obj_lists["aquamarine"]
    for aquamarine in aquamarines:
        if aquamarine.x > agent.x and (aquamarine.x - agent.x <= agent.y - aquamarine.y):
            return True
    return False


def is_amethyst_right_within_reach(env: Env) -> bool:
    """
    Checks whether an amethyst is to the right of the agent and within reach, considering
    that the amethyst moves down one cell per step and the agent can move one cell
    horizontally per step.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    amethysts = env.unwrapped.obj_lists["amethyst"]
    for amethyst in amethysts:
        if amethyst.x > agent.x and (amethyst.x - agent.x <= agent.y - amethyst.y):
            return True
    return False


def is_emerald_right_within_reach(env: Env) -> bool:
    """
    Checks whether an emerald is to the right of the agent and within reach, considering
    that the emerald moves down one cell per step and the agent can move one cell
    horizontally per step.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    emeralds = env.unwrapped.obj_lists["emerald"]
    for emerald in emeralds:
        if emerald.x > agent.x and (emerald.x - agent.x <= agent.y - emerald.y):
            return True
    return False


def is_rock_1_above(env: Env) -> bool:
    """
    Checks whether a rock is exactly 1 step above the agent, x-coordinates being equal, in
    the current state of the given GemCollector environment instance.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    rocks = env.unwrapped.obj_lists["rock"]
    for rock in rocks:
        if rock.x == agent.x and rock.y == agent.y - 1:
            return True
    return False


def is_wall_left(env: Env) -> bool:
    """
    Checks whether the agent is right next to a wall on its left.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    return agent.x == 0


def is_wall_right(env: Env) -> bool:
    """
    Checks whether the agent is right next to a wall on its right.
    """
    _env_validation(env)
    agent = env.unwrapped.agent
    return agent.x == 19


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
    concepts = [
        # Aquamarine concepts
        BinaryConcept(
            name="aquamarine_left",
            observation_presence_callback=is_aquamarine_left,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="aquamarine_right",
            observation_presence_callback=is_aquamarine_right,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="aquamarine_above",
            observation_presence_callback=is_aquamarine_above,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="aquamarine_left_within_reach",
            observation_presence_callback=is_aquamarine_left_within_reach,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="aquamarine_right_within_reach",
            observation_presence_callback=is_aquamarine_right_within_reach,
            environment_name=ENV_NAME,
        ),
        # Amethyst concepts
        BinaryConcept(
            name="amethyst_left",
            observation_presence_callback=is_amethyst_left,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="amethyst_right",
            observation_presence_callback=is_amethyst_right,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="amethyst_above",
            observation_presence_callback=is_amethyst_above,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="amethyst_left_within_reach",
            observation_presence_callback=is_amethyst_left_within_reach,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="amethyst_right_within_reach",
            observation_presence_callback=is_amethyst_right_within_reach,
            environment_name=ENV_NAME,
        ),
        # Emerald concepts
        BinaryConcept(
            name="emerald_left",
            observation_presence_callback=is_emerald_left,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="emerald_right",
            observation_presence_callback=is_emerald_right,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="emerald_above",
            observation_presence_callback=is_emerald_above,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="emerald_left_within_reach",
            observation_presence_callback=is_emerald_left_within_reach,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="emerald_right_within_reach",
            observation_presence_callback=is_emerald_right_within_reach,
            environment_name=ENV_NAME,
        ),
        # Other important concepts
        BinaryConcept(
            name="rock_1_above",
            observation_presence_callback=is_rock_1_above,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="lava_1_above",
            observation_presence_callback=is_lava_1_above,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="wall_left",
            observation_presence_callback=is_wall_left,
            environment_name=ENV_NAME,
        ),
        BinaryConcept(
            name="wall_right",
            observation_presence_callback=is_wall_right,
            environment_name=ENV_NAME,
        ),
    ]

    return concepts
