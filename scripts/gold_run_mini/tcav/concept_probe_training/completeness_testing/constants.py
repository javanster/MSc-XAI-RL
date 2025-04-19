from ...constants import MODEL_OF_INTEREST_NAME

CONCEPT_EXAMPLES_DIR_PATH = "rl_tcav_data/concept_examples/gold_run_mini/by_concept_functions/iter_2/model_of_interest_greedy_play/"
CONCEPT_PROBE_DIR_PATH = (
    f"rl_tcav_data/concept_probes/gold_run_mini/completeness_testing/{MODEL_OF_INTEREST_NAME}/"
)
MODEL_LAYERS_OF_INTEREST = [0, 1, 3, 4, 5]  # Not including flatten layer and output layer
CONCEPT_NAMES = [
    "gold_above",
    "gold_right",
    "gold_left",
    "gold_down",
    "green_exit_above",
    "green_exit_right",
    "green_exit_left",
    "green_exit_down",
    "purple_exit_above",
    "purple_exit_right",
    "purple_exit_left",
    "purple_exit_down",
    "lava_1_above",
    "lava_1_right",
    "lava_1_left",
    "lava_1_down",
    "wall_directly_above",
    "wall_directly_right",
    "wall_directly_left",
    "wall_directly_down",
    "random_binary",
]
CONCEPT_PREFIXES = [
    "binary_concept_gold_above",
    "binary_concept_gold_right",
    "binary_concept_gold_left",
    "binary_concept_gold_down",
    "binary_concept_green_exit_above",
    "binary_concept_green_exit_right",
    "binary_concept_green_exit_left",
    "binary_concept_green_exit_down",
    "binary_concept_purple_exit_above",
    "binary_concept_purple_exit_right",
    "binary_concept_purple_exit_left",
    "binary_concept_purple_exit_down",
    "binary_concept_lava_1_above",
    "binary_concept_lava_1_right",
    "binary_concept_lava_1_left",
    "binary_concept_lava_1_down",
    "binary_concept_wall_directly_above",
    "binary_concept_wall_directly_right",
    "binary_concept_wall_directly_left",
    "binary_concept_wall_directly_down",
    "binary_concept_random_binary",
]
