from pathlib import Path

import numpy
import deepdish as dd
import pandas as pd

from .regression import ols_regression

from .utils import sync_time_series


def _construct_eye_data(config, save_dataframe):
    eye_dataframe = []
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
            raw_data = data['sub-OFS_' + subject][session]['eye_features']
            # Need to orient and transpose,
            # because the data are of not same length
            raw_data['n_fixations'] = [raw_data['n_fixations']]
            raw_data['n_saccades'] = [raw_data['n_saccades']]
            raw_data['avg_pupil_size'] = [
                numpy.nanmean(raw_data['pupil_size'])
            ]

            temp_df = pd.DataFrame.from_dict(raw_data,
                                             orient='index').transpose()
            # Add additional information
            temp_df['subject'] = subject
            temp_df['complexity'] = session_name[session]

            # Append it the the global dataframe
            eye_dataframe.append(temp_df)

    eye_dataframe = pd.concat(eye_dataframe)

    if save_dataframe:
        save_path = Path(__file__).parents[2] / config['eye_features_path']
        eye_dataframe.to_hdf(save_path, key='eye_dataframe')

    return eye_dataframe


def _convert_to_map_coor(fixations, map_pos, window_size):
    width = int(window_size[0] / 2)
    height = int(window_size[1] / 2)
    for fixation, map_pos in zip(fixations, map_pos):
        fixation[3] = width - map_pos[0] + fixation[3]
        fixation[4] = height - map_pos[1] + fixation[4]
    return fixations


def _convert_to_global_coor(fixations, window_size):
    width = int(window_size[0] / 2)
    height = int(window_size[1] / 2)
    for fixation in fixations:
        fixation[3] = width + fixation[3]
        fixation[4] = height + fixation[4]
    return fixations


def eye_features_analysis(config, features):
    # Check is dataframe is already there
    read_path = Path(__file__).parents[2] / config['eye_features_path']
    if read_path.is_file():
        eye_dataframe = pd.read_hdf(read_path, key='eye_dataframe')
    else:
        # Generate the dataframe
        eye_dataframe = _construct_eye_data(config, save_dataframe=True)

    # Select the features
    eye_dataframe = eye_dataframe[features + ['complexity', 'subject']]
    eye_dataframe.dropna(inplace=True)
    eye_subject_group = eye_dataframe.groupby(['subject',
                                               'complexity']).count()

    eye_subject_group.reset_index(inplace=True)

    models = []
    for feature in features:
        models.append(ols_regression('complexity', feature, eye_subject_group))
    return models, eye_subject_group


def calculate_fixations(config, subject, session, in_map=False):
    subject_group = '/sub-OFS_' + subject
    read_path = Path(__file__).parents[2] / config['offset_features_path']
    data = dd.io.load(read_path, group=subject_group)

    eye_features = data[session]['eye_features']
    game_features = data[session]['game_features']

    # Fixation and fixation time stamps
    fixations = eye_features['fixations']
    eye_time_stamps = eye_features['time_stamps']
    # Convert from milli second to LSL time stamp
    fixation_time_stamps = [
        item[0] / 1000 + eye_time_stamps[0] for item in fixations
    ]

    # Get the map_pos and game time stamps
    map_pos = [item[0:2] for item in game_features['map_pos']]
    game_time_stamps = game_features['time_stamps']

    # Synchronize the time
    sync_index = sync_time_series(fixation_time_stamps,
                                  eye_features['time_kd_tree'],
                                  game_time_stamps,
                                  game_features['time_kd_tree'])
    map_pos = numpy.array(map_pos)[sync_index]

    # Convert to map co-ordinate systems
    if in_map:
        fixations = _convert_to_map_coor(fixations, map_pos, [1500, 750])
    else:
        fixations = _convert_to_global_coor(fixations, [1500, 750])

    return fixations
