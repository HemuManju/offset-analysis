from pathlib import Path

import numpy as np
import deepdish as dd
from scipy.spatial import KDTree

from .utils import _get_platoon_node_position, findkeys, _initial_nodes_setup


def _compute_visibility(states, complexity_states):
    raise NotImplementedError


def _get_shooting_nodes(states):
    node_index = _get_platoon_node_position(states)
    node_primitive = list(findkeys(states, 'primitive'))

    # Assert they are of same length
    assert len(node_index) == len(
        node_primitive), 'Data are of different length'

    return node_index, node_primitive


def _change_premitive(states, complexity_states):
    red_nodes = {
        '2': 'uav_p_1',
        '28': 'uav_p_2',
        '15': 'uav_p_3',
        '16': 'ugv_p_1',
        '1': 'ugv_p_2',
        '30': 'ugv_p_3',
    }
    node_index, node_primitive = _get_shooting_nodes(states)
    for idx, primitive in zip(node_index, node_primitive):
        if primitive == 'shooting':
            key = red_nodes[idx]
            vehicles_type = key.split('_')[0]
            complexity_states[i][vehicles_type][key]['primitive'] = 'shooting'
        else:
            complexity_states[i][vehicles_type][key]['primitive'] = 'formation'

    return complexity_states


def get_complexity_states(config, subject, session):
    # Read the states
    read_path = Path(__file__).parents[2] / config['offset_features_path']
    subject_group = '/sub-OFS_' + '/'.join([subject, session, 'game_features'])
    read_group = subject_group + '/states'
    states = dd.io.load(read_path, group=read_group)

    # Read the states
    read_path = Path(__file__).parents[2] / config['offset_features_path']
    subject_group = '/sub-OFS_' + '/'.join([subject, session, 'game_features'])
    read_group = subject_group + '/complexity_states'
    complex_states = dd.io.load(read_path, group=read_group)

    centroid_pos = np.array(list(findkeys(states, 'centroid_pos')), ndmin=2)
    n_ele = len(centroid_pos) - int(len(centroid_pos) // 6) * 6
    centroid_pos = centroid_pos[n_ele:].reshape(6, -1, order='F')
    test = np.abs(np.diff(centroid_pos, axis=1)).sum(axis=0)
    for t in test:
        print(t)
    # print(test)
    print(a)

    # first_index = next(i for i, j in enumerate(complex_states) if j)
    # print(first_index)
    # co_states = complex_states[first_index]['ugv']

    # source_pos = np.array(list(findkeys(co_states, 'centroid_pos')), ndmin=2)
    # # sink_pos = np.array(list(findkeys(co_states, 'sink_pos')), ndmin=2)

    # nodes_pos = _initial_nodes_setup()
    # nodes_pos = nodes_pos - nodes_pos[48, :]
    # nodes_kd_tree = KDTree(nodes_pos)

    # # Get node index and reshape to 6 by m
    # source_index = nodes_kd_tree.query(source_pos, k=1)[1]
    # # sink_index = nodes_kd_tree.query(sink_pos, k=1)[1]

    # print(source_index)

    # # Copy the states to complexity states
    # # complexity_states = states
    # # complexity_states = _change_premitive(states, complexity_states)