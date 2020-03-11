from pathlib import Path

import numpy
import deepdish as dd
import pandas as pd

from features.gaze.detectors import fixation_detection

from .regression import ols_regression

from .utils import sync_time_series


def _without_keys(dictionary, exclude):
    for i in exclude:
        dictionary.pop(i)
    return dictionary


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

    for subject in config['subjects']:
        for session in config['sessions']:
            read_group = ''.join(
                ['/sub-OFS_', subject, '/', session, '/eye_features'])
            raw_data = dd.io.load(read_path, group=read_group)
            raw_data = _without_keys(raw_data,
                                     ['time_kd_tree'])  # No need for KD tree

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


def _convert_eye_to_map_coor(positions, map_pos, window_size):
    width = int(window_size[0] / 2)
    height = int(window_size[1] / 2)
    positions[:, 0] = width - map_pos[:, 0] + positions[:, 0]
    positions[:, 1] = height - map_pos[:, 1] - positions[:, 1]
    return positions


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


def calculate_eye_movement(config, subject, session, in_map=False):
    subject_group = '/sub-OFS_' + subject + '/' + session
    read_path = Path(__file__).parents[2] / config['offset_features_path']
    eye_features = dd.io.load(read_path, group=subject_group + '/eye_features')
    game_features = dd.io.load(read_path,
                               group=subject_group + '/game_features')

    # Position and position time stamps
    positions = eye_features['pos']
    eye_time_stamps = eye_features['time_stamps']

    # Get the map_pos and game time stamps
    map_pos = [item[0:2] for item in game_features['map_pos']]
    map_pos = numpy.array(map_pos)[0:-1:2]
    game_time_stamps = game_features['time_stamps']

    # Synchronize the time
    sync_index = sync_time_series(eye_time_stamps,
                                  eye_features['time_kd_tree'],
                                  game_time_stamps,
                                  game_features['time_kd_tree'])
    positions = positions[sync_index, :]
    # Remove the nans
    drop_index = numpy.isnan(positions).any(axis=1)
    positions[drop_index] = [0, 0]
    map_pos[drop_index] = [0, 0]

    # Convert to map co-ordinate systems
    if in_map:
        positions = _convert_eye_to_map_coor(positions, map_pos, [1500, 750])
    else:
        positions = _convert_to_global_coor(positions, [1500, 750])

    return positions


def calculate_fixations(config, subject, session, in_map=False):
    subject_group = '/sub-OFS_' + subject + '/' + session
    read_path = Path(__file__).parents[2] / config['offset_features_path']
    eye_features = dd.io.load(read_path, group=subject_group + '/eye_features')
    game_features = dd.io.load(read_path,
                               group=subject_group + '/game_features')
    pos = eye_features['pos']
    index = ~numpy.isnan(pos).any(axis=1)
    pos = pos[index]

    # Fixation and fixation time stamps
    eye_time_stamps = eye_features['time_stamps']
    eye_time_stamps_milli = (eye_time_stamps - eye_time_stamps[0]) * 1000
    eye_time_stamps_milli = eye_time_stamps_milli[index]
    _, fixations = fixation_detection(pos[:, 0],
                                      pos[:, 1],
                                      eye_time_stamps_milli,
                                      mindur=100)

    # Convert from milli second to LSL time stamp:,0
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
