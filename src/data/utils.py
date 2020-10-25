import collections
import deepdish as dd


def nested_dict():
    return collections.defaultdict(nested_dict)


def save_dataset(path, dataset, save):
    """save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataset : dataset
        pytorch dataset.
    save : Bool

    """
    if save:
        dd.io.save(path, dataset, compression=('blosc', 5))

    return None


def compress_dataset(path):
    """compress the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataset : dataset
        pytorch dataset.
    save : Bool
    """

    dataset = dd.io.load(path)
    # New name
    file_name = path.split('.')
    file_name[-2] = file_name[-2] + '_compressed.'
    save_path = ''.join(file_name)
    dd.io.save(save_path, dataset, compression=('blosc', 5))

    return None


def read_dataset(path):
    """Read the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataset : dataset
        pytorch dataset.
    save : Bool

    """
    data = dd.io.load(path)
    return data
