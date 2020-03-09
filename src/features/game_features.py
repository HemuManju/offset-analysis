import numpy as np
from scipy.spatial import distance

from data.extract_data import read_xdf_game_data
from .utils import findkeys, _time_kd_tree


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


def _get_selected_node(game_epochs, node_position):
    user_input = list(findkeys(game_epochs, 'user_input'))
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


def _get_target_building(game_epochs, node_position):
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
    states = list(findkeys(game_epochs, 'state'))
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
        user_selected, _ = _get_selected_node(game_epochs, node_position)

    if idx in [38, 39, 40, 51]:
        target_id = idx
    else:
        target_id = user_selected[-1]
    return target_id


def _get_selected_platoons(game_epochs):
    selected_list = list(findkeys(game_epochs, 'selected'))
    selection_key = []
    for i, selected in enumerate(selected_list):
        if selected:
            key = 'uav_p_' if i <= 2 else 'ugv_p_'
            selection_key.append(key + str(i % 3 + 1))
    return selection_key


def _get_map_pos(game_epochs):
    map_pos = list(findkeys(game_epochs, 'map_pos'))
    return map_pos


def _get_states(game_epochs, complexity):
    if complexity:
        states = list(findkeys(game_epochs, 'complexity_state'))
    else:
        states = list(findkeys(game_epochs, 'state'))
    return states


def _get_casualities(game_epochs):
    causalities = []
    states = list(findkeys(game_epochs, 'state'))
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


def extract_game_features(config, subject, session):
    game_data = {}
    node_position = _initial_nodes_setup(config)

    # Read the game epochs
    game_epochs, time_stamps = read_xdf_game_data(config, subject, session)
    node_index, node_pos = _get_selected_node(game_epochs, node_position)
    game_data['selected_node'] = node_index
    game_data['selected_node_pos'] = node_pos
    game_data['time_stamps'] = time_stamps
    game_data['casualities'] = _get_casualities(game_epochs)
    game_data['platoon_selected'] = _get_selected_platoons(game_epochs)
    game_data['game_state'] = list(findkeys(game_epochs, 'game'))
    game_data['map_pos'] = _get_map_pos(game_epochs)
    game_data['complexity_states'] = _get_states(game_epochs, complexity=True)
    game_data['states'] = _get_states(game_epochs, complexity=False)
    game_data['target_building'] = _get_target_building(
        game_epochs, node_position)
    game_data['time_kd_tree'] = _time_kd_tree(np.array(time_stamps, ndmin=2).T)

    return game_data
