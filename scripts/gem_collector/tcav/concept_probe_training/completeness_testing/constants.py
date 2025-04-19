from ...constants import MODEL_OF_INTEREST_NAME

CONCEPT_EXAMPLES_DIR_PATH = "rl_tcav_data/concept_examples/gem_collector/by_concept_functions/iter_2/model_of_interest_greedy_play/"
CONCEPT_PROBE_DIR_PATH = (
    f"rl_tcav_data/concept_probes/gem_collector/completeness_testing/{MODEL_OF_INTEREST_NAME}/"
)
MODEL_LAYERS_OF_INTEREST = [0, 1, 3, 4, 5]  # Not including flatten layer and output layer
CONCEPT_NAMES = [
    "amethyst_above",
    "amethyst_left_within_reach",
    "amethyst_right_within_reach",
    "aquamarine_above",
    "aquamarine_left_within_reach",
    "aquamarine_right_within_reach",
    "emerald_above",
    "emerald_left_within_reach",
    "emerald_right_within_reach",
    "lava_1_above",
    "rock_1_above",
    "wall_left",
    "wall_right",
    "random_binary",
]
CONCEPT_PREFIXES = [
    "binary_concept_amethyst_above",
    "binary_concept_amethyst_left_within_reach",
    "binary_concept_amethyst_right_within_reach",
    "binary_concept_aquamarine_above",
    "binary_concept_aquamarine_left_within_reach",
    "binary_concept_aquamarine_right_within_reach",
    "binary_concept_emerald_above",
    "binary_concept_emerald_left_within_reach",
    "binary_concept_emerald_right_within_reach",
    "binary_concept_lava_1_above",
    "binary_concept_rock_1_above",
    "binary_concept_wall_left",
    "binary_concept_wall_right",
    "binary_concept_random_binary",
]
