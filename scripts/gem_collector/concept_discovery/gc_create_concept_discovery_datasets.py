import numpy as np

np.random.seed(28)

from utils import ensure_directory_exists


def _save_balanced_set(base_save_dir_path: str) -> None:
    target_class_left_examples = np.load(
        "rl_tcav_data/class_label_examples/gem_collector/model_of_interest/100000_examples_output_class_0.npy"
    )
    target_class_right_examples = np.load(
        "rl_tcav_data/class_label_examples/gem_collector/model_of_interest/100000_examples_output_class_1.npy"
    )
    target_class_do_nothing_examples = np.load(
        "rl_tcav_data/class_label_examples/gem_collector/model_of_interest/61498_examples_output_class_2.npy"
    )

    np.random.shuffle(target_class_left_examples)
    np.random.shuffle(target_class_right_examples)
    np.random.shuffle(target_class_do_nothing_examples)

    combined_set = np.array(
        [
            *target_class_left_examples[:10_000],
            *target_class_right_examples[:10_000],
            *target_class_do_nothing_examples[:10_000],
        ]
    )

    np.random.shuffle(combined_set)

    np.save(f"{base_save_dir_path}/target_class_balanced_30000_shuffled_examples.npy", combined_set)


def _save_do_nothing_set(base_save_dir_path: str) -> None:
    target_class_do_nothing_examples = np.load(
        "rl_tcav_data/class_label_examples/gem_collector/model_of_interest/61498_examples_output_class_2.npy"
    )

    np.random.shuffle(target_class_do_nothing_examples)

    target_class_do_nothing_examples = target_class_do_nothing_examples[:30_000]

    np.save(
        f"{base_save_dir_path}/target_class_do_nothing_30000_shuffled_examples.npy",
        target_class_do_nothing_examples,
    )


def _save_left_set(base_save_dir_path: str) -> None:
    target_class_left_examples = np.load(
        "rl_tcav_data/class_label_examples/gem_collector/model_of_interest/100000_examples_output_class_0.npy"
    )

    np.random.shuffle(target_class_left_examples)

    target_class_left_examples = target_class_left_examples[:30_000]

    np.save(
        f"{base_save_dir_path}/target_class_left_30000_shuffled_examples.npy",
        target_class_left_examples,
    )


def _save_right_set(base_save_dir_path: str) -> None:
    target_class_right_examples = np.load(
        "rl_tcav_data/class_label_examples/gem_collector/model_of_interest/100000_examples_output_class_1.npy"
    )

    np.random.shuffle(target_class_right_examples)

    target_class_right_examples = target_class_right_examples[:30_000]

    np.save(
        f"{base_save_dir_path}/target_class_right_30000_shuffled_examples.npy",
        target_class_right_examples,
    )


if __name__ == "__main__":
    # Generates files with shuffled target class examples, the class detemined by the actions chosen by model of interest
    # The intended use of these files are for concept discovery

    base_save_dir_path = "rl_concept_discovery_data/class_datasets_model_of_interest/gem_collector/"
    ensure_directory_exists(base_save_dir_path)

    _save_balanced_set(base_save_dir_path)
    _save_do_nothing_set(base_save_dir_path)
    _save_left_set(base_save_dir_path)
    _save_right_set(base_save_dir_path)
