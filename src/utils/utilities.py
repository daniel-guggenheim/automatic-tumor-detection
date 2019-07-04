import datetime
import pickle
from typing import Any


def now_time() -> str:
    return datetime.datetime.now().strftime('%H:%M:%S')


def load_pickle(file_path: str) -> Any:
    """
    Load a python variable from a pickle file.

    :param file_path: The path to the pickle file in which the variable was saved.
    :return: The python variable that was saved in the file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def save_pickle(my_variable: Any, file_path: str = 'pickle.p'):
    """
    Save a python variable to a pickle file.

    :param my_variable: The python variable to save (to retrieve it later)
    :param file_path: The path to the pickle file to create.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(my_variable, file)