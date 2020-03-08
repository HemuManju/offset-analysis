import itertools

import time

from pathlib import Path

import deepdish as dd
import pandas as pd
import numpy
import networkx as nx
from scipy import spatial
import matplotlib.pyplot as plt

from features.game_features import _initial_nodes_setup
from features.utils import findkeys

from visualization.graph_visualize import draw_graph

from .regression import ols_regression


def _construct_game_data(config, save_dataframe):
    game_dataframe = []
    session_name = {
        'S001': 'baseline',
        'S002': 'static_red',
        'S003': 'dynamic_red',
        'S004': 'static_red_smoke',
        'S005': 'dynamic_red_smoke'
    }

    read_path = Path(__file__).parents[2] / config['offset_features_path']
    data = dd.io.load(read_path)

    for subject in config['subjects']:
        for session in config['sessions']:
            raw_data = data['sub-OFS_' + subject][session]['game_features']
            # Need to orient and transpose,
            # because the data are of not same length
            temp_df = pd.DataFrame.from_dict(raw_data,
                                             orient='index').transpose()
            # Add additional information
            temp_df['subject'] = subject
            temp_df['complexity'] = session_name[session]

            # Append it the the global dataframe
            game_dataframe.append(temp_df)

    game_dataframe = pd.concat(game_dataframe)

    if save_dataframe:
        save_path = Path(__file__).parents[2] / config['game_features_path']
        game_dataframe.to_hdf(save_path, key='game_dataframe')

    return game_dataframe


def _construct_graph(config, target_id):
    read_path = Path(__file__).parents[2] / config['map_data_path']

    nodes_pos = numpy.genfromtxt(str(read_path) + '/nodes.csv',
                                 delimiter=',',
                                 usecols=(0, 1))
    nodes_pos = nodes_pos - nodes_pos[48, :]
    adjacency_matrix = numpy.genfromtxt(str(read_path) +
                                        '/nodes_adjacency_matrix.csv',
                                        delimiter=',')

    G = nx.from_numpy_matrix(adjacency_matrix)
    # Add node weight and color
    nx.set_node_attributes(G, name='node_weight', values=100)
    nx.set_node_attributes(G, name='node_color', values='#65A754')
    nx.set_node_attributes(G, name='position', values=0)

    # Buildings ids
    building_ids = [38, 39, 40, 48, 51]

    # Set node shape for target building
    if target_id in building_ids:
        G.nodes[target_id]['node_color'] = '#0E0E0E'

    for node, position in enumerate(nodes_pos):
        if node in building_ids:
            G.nodes[node]['node_weight'] = 1000
        else:
            G.nodes[node]['node_weight'] = 250
        G.nodes[node]['position'] = (position[0], -position[1])
    return G


def _get_user_actions(config, subject, session, use_resume=False):
    # Check is dataframe is already there
    subject_group = ''.join(
        ['/sub-OFS_', subject, '/', session, '/', 'game_features'])
    read_path = Path(__file__).parents[2] / config['offset_features_path']
    data = dd.io.load(read_path, group=subject_group)

    # Actions and game state
    selected_node = data['selected_node']

    if use_resume:
        resumes = data['resume_state']
        # Verify the length of data
        assert len(selected_node) == len(
            resumes), "Data are not of same length"

        # Sync it with resume time stamp
        actions = []
        for node, resume in zip(selected_node, resumes):
            if resume:
                actions.append(node)
    else:
        actions = selected_node

    return actions


def _get_platoon_node_position(config, subject, session, complexity=False):
    if complexity:
        subject_group = ''.join([
            '/sub-OFS_', subject, '/', session, '/game_features',
            '/complexity_states'
        ])
    else:
        subject_group = ''.join(
            ['/sub-OFS_', subject, '/', session, '/game_features', '/states'])

    read_path = Path(__file__).parents[2] / config['offset_features_path']
    states = dd.io.load(read_path, group=subject_group)

    # Node position
    nodes_pos = _initial_nodes_setup(config)
    nodes_pos = nodes_pos - nodes_pos[48, :]
    nodes_kd_tree = spatial.KDTree(nodes_pos)

    centroid_pos = numpy.array(list(findkeys(states, 'centroid_pos')), ndmin=2)
    # Get node index and reshape to 6 by m
    node_index = nodes_kd_tree.query(centroid_pos, k=1)[1]
    n_ele = len(node_index) - int(len(node_index) // 6) * 6
    node_index = node_index[n_ele:].reshape(6, -1, order='F')

    return node_index


def game_features_analysis(config, features):
    # Check is dataframe is already there
    read_path = Path(__file__).parents[2] / config['game_features_path']
    if read_path.is_file():
        game_dataframe = pd.read_hdf(read_path, key='game_dataframe')
    else:
        # Generate the dataframe
        game_dataframe = _construct_game_data(config, save_dataframe=True)

    # Select the features and clean
    game_dataframe = game_dataframe[features + ['complexity', 'subject']]
    game_dataframe.dropna(inplace=True)
    game_subject_group = game_dataframe.groupby(['subject',
                                                 'complexity']).count()
    game_subject_group.reset_index(inplace=True)

    models = []
    for feature in features:
        models.append(ols_regression('complexity', feature,
                                     game_subject_group))
    return models, game_subject_group


def graph_with_user_actions(config, subject, session):
    selected_nodes = _get_user_actions(config, subject, session)
    complexity_nodes = _get_platoon_node_position(config,
                                                  subject,
                                                  session,
                                                  complexity=True)
    platoon_nodes = _get_platoon_node_position(config, subject, session)

    # Get the target building
    subject_group = ''.join([
        '/sub-OFS_', subject, '/', session, '/',
        'game_features/target_building'
    ])
    read_path = Path(__file__).parents[2] / config['offset_features_path']
    target_id = dd.io.load(read_path, group=subject_group)

    # # Verify the length of data
    # assert len(selected_nodes
    #            ) == platoon_nodes.shape[1], "Data are not of same length"

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(0, len(selected_nodes), 20):
        # Get the graph
        G = _construct_graph(config, target_id)

        # Action node
        action_node = selected_nodes[i]
        if action_node:
            G.nodes[action_node]['node_color'] = '#8E539F'

        # Platoon node
        platoon_node = platoon_nodes[:, i].tolist()
        complexity_node = complexity_nodes[:, i].tolist()
        for node, complex_node in zip(platoon_node, complexity_node):
            G.nodes[node]['node_color'] = '#4A7DB3'
            G.nodes[node]['node_weight'] = 500
            G.nodes[complex_node]['node_color'] = '#D2352B'
            G.nodes[complex_node]['node_weight'] = 500

        draw_graph(G, ax)
        plt.pause(1e-1)
        ax.cla()
    return None
