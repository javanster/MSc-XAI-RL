import numpy as np

np.random.seed(28)

from utils import ensure_directory_exists


def _save_balanced_set(base_save_dir_path: str) -> None:
    target_class_up_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/7070_examples_output_class_0.npy"
    )
    target_class_upright_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/34880_examples_output_class_1.npy"
    )
    target_class_right_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/22997_examples_output_class_2.npy"
    )
    target_class_downright_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/11686_examples_output_class_3.npy"
    )
    target_class_down_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/9680_examples_output_class_4.npy"
    )
    target_class_downleft_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/7738_examples_output_class_5.npy"
    )
    target_class_left_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/12383_examples_output_class_6.npy"
    )
    target_class_upleft_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/24696_examples_output_class_7.npy"
    )

    np.random.shuffle(target_class_up_examples)
    np.random.shuffle(target_class_upright_examples)
    np.random.shuffle(target_class_right_examples)
    np.random.shuffle(target_class_downright_examples)
    np.random.shuffle(target_class_down_examples)
    np.random.shuffle(target_class_downleft_examples)
    np.random.shuffle(target_class_left_examples)
    np.random.shuffle(target_class_upleft_examples)

    # Combines to a set of length 30_000
    combined_set = np.array(
        [
            *target_class_up_examples[:3_750],
            *target_class_upright_examples[:3_750],
            *target_class_right_examples[:3_750],
            *target_class_downright_examples[:3_750],
            *target_class_down_examples[:3_750],
            *target_class_downleft_examples[:3_750],
            *target_class_left_examples[:3_750],
            *target_class_upleft_examples[:3_750],
        ]
    )

    np.random.shuffle(combined_set)

    np.save(
        f"{base_save_dir_path}/target_class_balanced_{len(combined_set)}_shuffled_examples.npy",
        combined_set,
    )


def _save_up_set(base_save_dir_path: str) -> None:
    target_class_up_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/7070_examples_output_class_0.npy"
    )

    np.random.shuffle(target_class_up_examples)

    np.save(
        f"{base_save_dir_path}/target_class_up_{len(target_class_up_examples)}_shuffled_examples.npy",
        target_class_up_examples,
    )


def _save_upright_set(base_save_dir_path: str) -> None:
    target_class_upright_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/34880_examples_output_class_1.npy"
    )

    np.random.shuffle(target_class_upright_examples)
    target_class_upright_examples = target_class_upright_examples[:30_000]

    np.save(
        f"{base_save_dir_path}/target_class_upright_{len(target_class_upright_examples)}_shuffled_examples.npy",
        target_class_upright_examples,
    )


def _save_right_set(base_save_dir_path: str) -> None:
    target_class_right_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/22997_examples_output_class_2.npy"
    )

    np.random.shuffle(target_class_right_examples)

    np.save(
        f"{base_save_dir_path}/target_class_right_{len(target_class_right_examples)}_shuffled_examples.npy",
        target_class_right_examples,
    )


def _save_downright_set(base_save_dir_path: str) -> None:
    target_class_downright_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/11686_examples_output_class_3.npy"
    )

    np.random.shuffle(target_class_downright_examples)

    np.save(
        f"{base_save_dir_path}/target_class_downright_{len(target_class_downright_examples)}_shuffled_examples.npy",
        target_class_downright_examples,
    )


def _save_down_set(base_save_dir_path: str) -> None:
    target_class_down_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/9680_examples_output_class_4.npy"
    )

    np.random.shuffle(target_class_down_examples)

    np.save(
        f"{base_save_dir_path}/target_class_down_{len(target_class_down_examples)}_shuffled_examples.npy",
        target_class_down_examples,
    )


def _save_downleft_set(base_save_dir_path: str) -> None:
    target_class_downleft_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/7738_examples_output_class_5.npy"
    )

    np.random.shuffle(target_class_downleft_examples)

    np.save(
        f"{base_save_dir_path}/target_class_downleft_{len(target_class_downleft_examples)}_shuffled_examples.npy",
        target_class_downleft_examples,
    )


def _save_left_set(base_save_dir_path: str) -> None:
    target_class_left_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/12383_examples_output_class_6.npy"
    )

    np.random.shuffle(target_class_left_examples)

    np.save(
        f"{base_save_dir_path}/target_class_left_{len(target_class_left_examples)}_shuffled_examples.npy",
        target_class_left_examples,
    )


def _save_upleft_set(base_save_dir_path: str) -> None:
    target_class_upleft_examples = np.load(
        "rl_tcav_data/class_label_examples/minecart_counter/model_of_interest/24696_examples_output_class_7.npy"
    )

    np.random.shuffle(target_class_upleft_examples)

    np.save(
        f"{base_save_dir_path}/target_class_upleft_{len(target_class_upleft_examples)}_shuffled_examples.npy",
        target_class_upleft_examples,
    )


if __name__ == "__main__":
    # Generates files with shuffled target class examples, the class detemined by the actions chosen by model of interest
    # The intended use of these files are for concept discovery

    base_save_dir_path = (
        "rl_concept_discovery_data/class_datasets_model_of_interest/minecart_counter/"
    )
    ensure_directory_exists(base_save_dir_path)

    _save_balanced_set(base_save_dir_path)
    _save_up_set(base_save_dir_path)
    _save_upright_set(base_save_dir_path)
    _save_right_set(base_save_dir_path)
    _save_downright_set(base_save_dir_path)
    _save_down_set(base_save_dir_path)
    _save_downleft_set(base_save_dir_path)
    _save_left_set(base_save_dir_path)
    _save_upleft_set(base_save_dir_path)
