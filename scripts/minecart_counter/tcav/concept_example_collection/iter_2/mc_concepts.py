import numpy as np
from gymnasium import Env

from rl_tcav import BinaryConcept, ContinuousConcept

from .constants import ENV_NAME


def _env_validation(env: Env) -> None:
    if env.unwrapped.name != ENV_NAME:
        raise ValueError("Incorrect env provided. Must be an instance of the MinecartCounter env")


def minecarts_n(env: Env) -> float:
    _env_validation(env)
    return float(len(env.unwrapped.minecarts))


# Direction vectors: (dy, dx)
DIRECTION_OFFSETS = {
    "up": (-1, 0),
    "up_right": (-1, 1),
    "right": (0, 1),
    "down_right": (1, 1),
    "down": (1, 0),
    "down_left": (1, -1),
    "left": (0, -1),
    "up_left": (-1, -1),
}


def _is_entity_in_direction(env: Env, entities, dx: int, dy: int) -> bool:
    agent = env.unwrapped.agent
    target_pos = (agent.x + dx, agent.y + dy)
    for ent in entities:
        if (ent.x, ent.y) == target_pos:
            return True
    return False


# Creates one function for each of the 8 directions, checking whether a minecart is 1 step away from the agent in the respective direction
def generate_minecart_checks():
    def make_fn(direction: str, dx: int, dy: int):
        def fn(env: Env) -> bool:
            _env_validation(env)
            return _is_entity_in_direction(env, env.unwrapped.minecarts, dx, dy)

        fn.__name__ = f"is_minecart_1_{direction}"
        return fn

    return [make_fn(dir_name, dx, dy) for dir_name, (dy, dx) in DIRECTION_OFFSETS.items()]


# Creates one function for each of the 8 directions, checking whether a wall is 1 step away from the agent in the respective direction
def generate_wall_checks():
    def make_fn(direction: str, dx: int, dy: int):
        def fn(env: Env) -> bool:
            _env_validation(env)
            return _is_entity_in_direction(env, env.unwrapped.walls, dx, dy)

        fn.__name__ = f"is_wall_1_{direction}"
        return fn

    return [make_fn(dir_name, dx, dy) for dir_name, (dy, dx) in DIRECTION_OFFSETS.items()]


def generate_goal_direction_binary_fns():
    def make_fn(direction: str, check_fn, goal_index: int):
        def fn(env: Env) -> bool:
            _env_validation(env)
            agent = env.unwrapped.agent
            goal = env.unwrapped.goals[goal_index - 1]
            return check_fn(agent, goal)

        fn.__name__ = f"is_goal_{goal_index}_{direction}"
        return fn

    def is_up(agent, goal):
        return goal.y < agent.y

    def is_down(agent, goal):
        return goal.y > agent.y

    def is_left(agent, goal):
        return goal.x < agent.x

    def is_right(agent, goal):
        return goal.x > agent.x

    check_map = {
        "up": is_up,
        "right": is_right,
        "down": is_down,
        "left": is_left,
    }

    fns = []
    for goal_index in range(1, 9):
        for direction, check_fn in check_map.items():
            fns.append(make_fn(direction, check_fn, goal_index))
    return fns


def get_mc_continuous_concepts():
    continuous_concepts = [
        ContinuousConcept(
            name="minecarts_n",
            observation_presence_callback=minecarts_n,
            environment_name=ENV_NAME,
        )
    ]
    return continuous_concepts


def get_mc_binary_concepts():
    binary_concepts = []

    for direction_fn in generate_minecart_checks():
        binary_concepts.append(
            BinaryConcept(
                name=direction_fn.__name__.replace("is_", ""),
                observation_presence_callback=direction_fn,
                environment_name=ENV_NAME,
            )
        )

    for direction_fn in generate_wall_checks():
        binary_concepts.append(
            BinaryConcept(
                name=direction_fn.__name__.replace("is_", ""),
                observation_presence_callback=direction_fn,
                environment_name=ENV_NAME,
            )
        )

    for direction_fn in generate_goal_direction_binary_fns():
        binary_concepts.append(
            BinaryConcept(
                name=direction_fn.__name__.replace("is_", ""),
                observation_presence_callback=direction_fn,
                environment_name=ENV_NAME,
            )
        )

    return binary_concepts
