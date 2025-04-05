import numpy as np

np.random.seed(28)

from utils import ensure_directory_exists


def _save_balanced_set(base_save_dir_path: str) -> None:
    target_class_up_examples = np.load(
        "rl_tcav_data/class_label_examples/gold_run_mini/model_of_interest/1725_examples_output_class_0.npy"
    )
    target_class_right_examples = np.load(
        "rl_tcav_data/class_label_examples/gold_run_mini/model_of_interest/1857_examples_output_class_1.npy"
    )
    target_class_down_examples = np.load(
        "rl_tcav_data/class_label_examples/gold_run_mini/model_of_interest/1872_examples_output_class_2.npy"
    )
    target_class_left_examples = np.load(
        "rl_tcav_data/class_label_examples/gold_run_mini/model_of_interest/1818_examples_output_class_3.npy"
    )

    np.random.shuffle(target_class_up_examples)
    np.random.shuffle(target_class_right_examples)
    np.random.shuffle(target_class_down_examples)
    np.random.shuffle(target_class_left_examples)

    n_examples_from_each_class = min(
        len(target_class_up_examples),
        len(target_class_right_examples),
        len(target_class_down_examples),
        len(target_class_left_examples),
    )

    combined_set = np.array(
        [
            *target_class_up_examples[:n_examples_from_each_class],
            *target_class_right_examples[:n_examples_from_each_class],
            *target_class_down_examples[:n_examples_from_each_class],
            *target_class_left_examples[:n_examples_from_each_class],
        ]
    )

    np.random.shuffle(combined_set)

    np.save(
        f"{base_save_dir_path}/target_class_balanced_{len(combined_set)}_shuffled_examples.npy",
        combined_set,
    )


def _save_up_set(base_save_dir_path: str) -> None:
    target_class_up_examples = np.load(
        "rl_tcav_data/class_label_examples/gold_run_mini/model_of_interest/1725_examples_output_class_0.npy"
    )

    np.random.shuffle(target_class_up_examples)

    np.save(
        f"{base_save_dir_path}/target_class_up_{len(target_class_up_examples)}_shuffled_examples.npy",
        target_class_up_examples,
    )


def _save_right_set(base_save_dir_path: str) -> None:
    target_class_right_examples = np.load(
        "rl_tcav_data/class_label_examples/gold_run_mini/model_of_interest/1857_examples_output_class_1.npy"
    )

    np.random.shuffle(target_class_right_examples)

    np.save(
        f"{base_save_dir_path}/target_class_right_{len(target_class_right_examples)}_shuffled_examples.npy",
        target_class_right_examples,
    )


def _save_down_set(base_save_dir_path: str) -> None:

    target_class_down_examples = np.load(
        "rl_tcav_data/class_label_examples/gold_run_mini/model_of_interest/1872_examples_output_class_2.npy"
    )

    np.save(
        f"{base_save_dir_path}/target_class_down_{len(target_class_down_examples)}_shuffled_examples.npy",
        target_class_down_examples,
    )

    np.random.shuffle(target_class_down_examples)


def _save_left_set(base_save_dir_path: str) -> None:
    target_class_left_examples = np.load(
        "rl_tcav_data/class_label_examples/gold_run_mini/model_of_interest/1818_examples_output_class_3.npy"
    )

    np.random.shuffle(target_class_left_examples)

    np.save(
        f"{base_save_dir_path}/target_class_left_{len(target_class_left_examples)}_shuffled_examples.npy",
        target_class_left_examples,
    )


if __name__ == "__main__":
    # Generates files with shuffled target class examples, the class detemined by the actions chosen by model of interest
    # The intended use of these files are for concept discovery

    base_save_dir_path = "rl_concept_discovery_data/class_datasets_model_of_interest/gold_run_mini/"
    ensure_directory_exists(base_save_dir_path)

    _save_balanced_set(base_save_dir_path)
    _save_up_set(base_save_dir_path)
    _save_right_set(base_save_dir_path)
    _save_down_set(base_save_dir_path)
    _save_left_set(base_save_dir_path)
