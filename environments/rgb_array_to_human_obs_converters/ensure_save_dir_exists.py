import os


def ensure_save_directory_exists(directory_path: str) -> None:
    """
    Ensure that the save directory exists.

    If the specified directory does not exist, it is created.

    Parameters
    ----------
    directory_path : str
        The path to the directory to check or create.
    """
    directory = os.path.dirname(directory_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
