from pathlib import Path

import numpy
import deepdish as dd
import pandas as pd

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


def get_user_actions(config, subject, session):
    # Check is dataframe is already there
    subject_group = ''.join(
        ['/sub-OFS_', subject, '/', session, '/', 'game_features'])
    read_path = Path(__file__).parents[2] / config['offset_features_path']
    data = dd.io.load(read_path, group=subject_group)

    print(data['selected_node'])
