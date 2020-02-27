import numpy as np
from scipy.spatial import distance

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


def _get_selected_platoons(game_data):
    selected_list = list(findkeys(game_data, 'selected'))
    selection_key = []
    for i, selected in enumerate(selected_list):
        if selected:
            key = 'uav_p_' if i <= 2 else 'ugv_p_'
            selection_key.append(key + str(i % 3 + 1))
    return selection_key


def _get_casualities(game_data):
    causalities = []
    for i in range(3):
        # Extract UAV and UGV group
        uav_group = list(findkeys(game_data, 'uav_p_' + str(i + 1)))
        uav_casulaties = list(findkeys(uav_group, 'casualities'))

        causalities.append(len(uav_casulaties[-1]))

        ugv_group = list(findkeys(game_data, 'ugv_p_' + str(i + 1)))
        ugv_casulaties = list(findkeys(ugv_group, 'casualities'))

        causalities.append(len(ugv_casulaties[-1]))
    return causalities


def extract_game_features(config, subject, session):
    game_data = {}
    node_position = _initial_nodes_setup(config)

    game_epochs, time_stamps = read_xdf_game_data(config, subject, session)

    node_index, node_pos = _get_selected_node(game_epochs, node_position)
    game_data['selected_node'] = node_index
    game_data['casualities'] = _get_casualities(game_epochs)
    game_data['selected_node_pos'] = node_pos
    game_data['platoon_selected'] = _get_selected_platoons(game_epochs)
    game_data['pause'] = list(findkeys(game_epochs, 'pause'))
    game_data['resume_state'] = list(findkeys(game_epochs, 'resume'))
    game_data['time_stamps'] = time_stamps

    return game_data