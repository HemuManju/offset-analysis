from pathlib import Path

import numpy as np
import deepdish as dd
from scipy.spatial import distance, KDTree

from data.extract_data import read_xdf_game_data

from .utils import findkeys


def _initial_nodes_setup(config):
    """Performs initial nodes setup
    """
    # Nodes setup
    path = config['map_data_path'] + 'nodes.csv'
    position_data = np.genfromtxt(path, delimiter=',', usecols=[0, 1])
    for i in range(config['n_nodes']):
        position_data[i, :] = [
            position_data[i][1] * 1.125, position_data[i][0] / 1.125
        ]
    return position_data


def _get_selected_node(game_data, node_position):
    user_input = list(findkeys(game_data, 'user_input'))
    target_pos = list(findkeys(user_input, 'target_pos'))
    node_index = []
    node_pos = []
    for point in target_pos:
        if point:
            pos = np.array([(point[1] - 340) * 0.2, (point[0] - 420) * 0.2],
                           ndmin=2)
            closet_index = distance.cdist(node_position, pos).argmin()
            node_index.append(closet_index)
            node_pos.append(pos)
        else:
            node_index.append([])
            node_pos.append([])
    return node_index, node_pos


def _get_target_building(game_data, node_position):
    target_buildings = {
        0: 38,
        1: 39,
        2: 40,
        3: 51,
    }
    target_pos = np.array(
        [[57.91725, 149.91111111], [38.48962, 96.25777778],
         [21.99375, 189.26222222], [-40.32225, 178.85333333]],
        ndmin=2)
    # Find the platoon centroid
    states = list(findkeys(game_data, 'state'))
    ugv_group = list(findkeys(states, 'ugv'))
    ugv_centroid = list(findkeys(ugv_group, 'centroid_pos'))
    centroid_pos = np.array(ugv_centroid, ndmin=2)

    # Find the index of distance below the threshold
    dist = distance.cdist(target_pos, centroid_pos)[:]
    n_ele = len(dist) - int(len(dist) // 12) * 12
    dist = dist[n_ele:].reshape(12, -1, order='F')
    row_idx, column_idx = np.where(dist < 8)
    try:
        time_stamp = np.argmax(column_idx)
        idx = target_buildings[int(row_idx[time_stamp] % 4)]
    except ValueError:
        idx = 0
        user_selected, _ = _get_selected_node(game_data, node_position)

    if idx in [38, 39, 40, 51]:
        target_id = idx
    else:
        target_id = user_selected[-1]
    return target_id


def _get_selected_platoons(game_data):
    selected_list = list(findkeys(game_data, 'selected'))
    selection_key = []
    for i, selected in enumerate(selected_list):
        if selected:
            key = 'uav_p_' if i <= 2 else 'ugv_p_'
            selection_key.append(key + str(i % 3 + 1))
    return selection_key


def _get_map_pos(game_data):
    map_pos = list(findkeys(game_data, 'map_pos'))
    return map_pos


def _get_states(game_data, complexity, session=None):
    if complexity:
        states = list(findkeys(game_data, 'complexity_state'))
    else:
        states = list(findkeys(game_data, 'state'))
    return states


def _get_casualities(game_data):
    causalities = []
    states = list(findkeys(game_data, 'state'))
    if states:
        # Extract UAV and UGV group
        uav_group = list(findkeys(states[-1], 'uav'))
        uav_casulaties = list(findkeys(uav_group, 'casualities'))
        causalities.append([len(i) for i in uav_casulaties])

        ugv_group = list(findkeys(states[-1], 'ugv'))
        ugv_casulaties = list(findkeys(ugv_group, 'casualities'))
        causalities.append([len(i) for i in ugv_casulaties])
    else:
        causalities = [[0] * 3, [0] * 3]  # 3 UGV and 3 UAV
    return causalities


def _read_game_data(config, subject, session):
    read_path = Path(__file__).parents[2] / config['offset_features_path']
    read_group = '/sub-OFS_' + '/'.join([subject, session, 'game_features'])

    # Read game data
    game_data = dd.io.load(read_path, group=read_group)
    return game_data


def _get_complexity_node_index(config, game_data):
    # Get the complexity states
    co_states = game_data['complexity_states']

    if co_states:
        centroid_pos = np.array(list(findkeys(co_states, 'centroid_pos')),
                                ndmin=2)
    else:
        temp = game_data['states']  # Copy the states themselves
        centroid_pos = np.array(list(findkeys(temp, 'centroid_pos')), ndmin=2)

    # Node position
    nodes_pos = _initial_nodes_setup(config)
    nodes_pos = nodes_pos - nodes_pos[48, :]
    nodes_kd_tree = KDTree(nodes_pos)

    # Get node index and reshape to 6 by m
    node_index = nodes_kd_tree.query(centroid_pos, k=1)[1]
    n_ele = len(node_index) - int(len(node_index) // 6) * 6
    node_index = node_index[n_ele:].reshape(6, -1, order='F')
    return node_index


def _extract_action_info(game_data):

    # Find pauses and resumes
    game_state = game_data['game_state']
    pauses = list(findkeys(game_state, 'pause'))
    resumes = list(findkeys(game_state, 'resume'))

    # Get game time stamps
    game_time_stamps = game_data['time_stamps']
    pause_time_stamps, resume_time_stamps = [], []

    # TODO: Need to implement in a robust way
    # The last element is not pause
    for i, (pause, resume) in enumerate(zip(pauses, resumes)):
        if pause and not pauses[i - 1]:
            pause_time_stamps.append(game_time_stamps[i])
        if pause and not pauses[i + 1]:
            resume_time_stamps.append(game_time_stamps[i])
    assert len(pause_time_stamps) == len(
        resume_time_stamps), 'Length is different'

    # Find the difference between pause and resume
    epoch_lengths = [
        rt - pt for rt, pt in zip(resume_time_stamps, pause_time_stamps)
    ]

    # Return epoch length and pauses
    return pause_time_stamps, resume_time_stamps, epoch_lengths


def compute_option_type(config, subject, session):
    # Read the game data
    game_data = _read_game_data(config, subject, session)

    # Target nodes
    target_nodes = [38, 39, 40, 51]

    # Complexity nodes
    complexity_nodes = _get_complexity_node_index(config, game_data)

    # Pause and resume time stamps
    pause_time_stamps, _, _ = _extract_action_info(game_data)

    # User actions
    user_selected_nodes = game_data['selected_node']

    # Option used by the subjects
    option_type = []
    epoch_time = []
    game_time_stamps = game_data['time_stamps'].tolist()
    for time_stamp in pause_time_stamps:
        # User selection after the resume
        time_id = game_time_stamps.index(time_stamp) + 1
        epoch_time.append(game_time_stamps[time_id])
        selected_node = user_selected_nodes[time_id]

        # Check with three cases
        if selected_node in target_nodes:
            option_type.append('target_option')
        elif selected_node in complexity_nodes[:, time_id]:
            option_type.append('engage_option')
        else:
            option_type.append('caution_option')

    return option_type, epoch_time


def extract_game_features(config, subject, session):
    game_features = {}
    node_position = _initial_nodes_setup(config)

    # Read the game epochs
    game_data, time_stamps = read_xdf_game_data(config, subject, session)
    node_index, node_pos = _get_selected_node(game_data, node_position)
    game_features['selected_node'] = node_index
    game_features['selected_node_pos'] = node_pos
    game_features['time_stamps'] = time_stamps
    game_features['casualities'] = _get_casualities(game_data)
    game_features['platoon_selected'] = _get_selected_platoons(game_data)
    game_features['game_state'] = list(findkeys(game_data, 'game'))
    game_features['map_pos'] = _get_map_pos(game_data)
    game_features['complexity_states'] = _get_states(game_data,
                                                     complexity=True,
                                                     session=session)
    return game_features
