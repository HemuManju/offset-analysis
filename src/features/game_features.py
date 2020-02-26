import collections
from pathlib import Path

import numpy as np
import pandas as pd
import deepdish as dd

from scipy.spatial import distance

from .utils import findkeys
from data.utils import read_dataset


def individual_features_with_time(config):
    session_name = {
        'S001': 'baseline',
        'S002': 'static_red',
        'S003': 'dynamic_red',
        'S004': 'static_red_smoke',
        'S005': 'dynamic_red_smoke'
    }
    column_names = [
        'mot', 'vs', 'completion_time', 'subject', 'complexity', 'distance'
    ]
    individual_diff_df = pd.DataFrame(np.empty((0, len(column_names))),
                                      columns=column_names)

    # Game data
    read_path = Path(__file__).parents[2] / config['game_features_path']
    game_data = read_dataset(str(read_path))

    # Read path
    read_path = Path(__file__).parents[2] / config['raw_offset_dataset']

    # Read the file
    df = []
    for subject in config['subjects']:
        individual_diff = dd.io.load(str(read_path),
                                     group='/sub_OFS_' + subject +
                                     '/individual_difference/')
        for session in config['sessions']:
            group = '/sub_OFS_' + '/'.join(
                [subject, session, 'eeg', 'time_stamps'])
            eeg_time = dd.io.load(str(read_path), group=group)
            duration = eeg_time[-1] - eeg_time[0]
            temp_data = game_data['sub_OFS_' + subject][session]['data']
            distance = np.linalg.norm(temp_data['selected_node_pos'][-1])

            data = [
                individual_diff['mot'], individual_diff['vs'], duration,
                subject, session_name[session], distance
            ]
            df.append(data)

    individual_diff_df = pd.DataFrame(df, columns=column_names)

    # Clean before return
    individual_diff_df.dropna()
    return individual_diff_df


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
    selected_list = list(findkeys(game_data, 'casualities'))
    print(selected_list)
    return None


def _calculate_game_features(game_data, node_position):
    data = {}
    node_index, node_pos = _get_selected_node(game_data, node_position)
    data['selected_node'] = node_index
    data['selected_node_pos'] = node_pos
    data['platoon_selected'] = _get_selected_platoons(game_data)
    data['pause'] = list(findkeys(game_data, 'pause'))
    data['resume_state'] = list(findkeys(game_data, 'resume'))
    return data


def extract_game_features(config):
    position_data = _initial_nodes_setup(config)
    game_dataset = {}

    # Reading path
    read_path = Path(__file__).parents[2] / config['raw_offset_dataset']

    for subject in config['subjects']:
        data = collections.defaultdict(dict)
        for session in config['sessions']:
            # Read only the game data
            group = '/sub_OFS_' + '/'.join([subject, session, 'game'])
            game_data = dd.io.load(str(read_path), group=group)
            data[session]['data'] = _calculate_game_features(
                game_data['data'], position_data)
            data[session]['time_stamps'] = game_data['time_stamps']
        game_dataset['sub_OFS_' + subject] = data
    return game_dataset
