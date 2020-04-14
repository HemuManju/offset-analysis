import collections

from scipy.stats import gamma
from scipy import spatial


def findkeys(var, key):
    if isinstance(var, dict):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, (dict, list)):
                yield from findkeys(v, key)
    elif isinstance(var, list):
        for d in var:
            yield from findkeys(d, key)


def nested_dict():
    return collections.defaultdict(nested_dict)


def _time_kd_tree(time_stamps):
    time_tree = spatial.KDTree(time_stamps)
    return time_tree


def gamma_percentile(data, percentile):
    """Fit recinormal distribution and get the values at given percentile values.

    Parameters
    ----------
    data : array
        Numpy array.
    percentile : list
        Percentile list.

    Returns
    -------
    array
        reaction time at given percentile.

    """
    result = gamma.fit(data)
    time_th = gamma.ppf(percentile,
                        a=result[0],
                        loc=result[1],
                        scale=result[2])
    return time_th
