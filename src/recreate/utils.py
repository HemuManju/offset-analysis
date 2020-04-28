from pathlib import Path
import yaml

import numpy as np
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


def _initial_nodes_setup():
    """Performs initial nodes setup
    """
    # The configuration file
    config_path = Path(__file__).parents[1] / 'config.yml'
    config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

    # Nodes setup
    path = config['map_data_path'] + 'nodes.csv'
    position_data = np.genfromtxt(path, delimiter=',', usecols=[0, 1])
    for i in range(config['n_nodes']):
        position_data[i, :] = [
            position_data[i][1] * 1.125, position_data[i][0] / 1.125
        ]
    return position_data


def _get_platoon_node_position(states):

    # Node position
    nodes_pos = _initial_nodes_setup()
    nodes_pos = nodes_pos - nodes_pos[48, :]
    nodes_kd_tree = spatial.KDTree(nodes_pos)

    centroid_pos = np.array(list(findkeys(states, 'centroid_pos')), ndmin=2)

    # Get node index and reshape to 6 by m
    node_index = nodes_kd_tree.query(centroid_pos, k=1)[1]
    return node_index
