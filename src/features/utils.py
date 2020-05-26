from pathlib import Path
import collections
import deepdish as dd

import numpy as np
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


def find_nearest_time_stamp(time_kd_tree, query_time):
    query_time = np.array(query_time, ndmin=2)
    nearest_t_id = time_kd_tree.query(query_time, k=1)[1]

    # Save index and value in a dictionary
    nearest_t_stamp = {}
    nearest_t_stamp['time'] = time_kd_tree.data[nearest_t_id].flatten()[0]
    nearest_t_stamp['id'] = nearest_t_id[0]
    return nearest_t_stamp


def nested_dict():
    return collections.defaultdict(nested_dict)


def construct_time_kd_tree(time_stamps):
    time_tree = spatial.KDTree(time_stamps)
    return time_tree


def read_data(config, subject, session, features):
    read_path = Path(__file__).parents[2] / config['offset_features_path']
    read_group = '/sub-OFS_' + '/'.join([subject, session, features])

    # Read game data
    data = dd.io.load(read_path, group=read_group)
    return data


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
