from pathlib import Path

import deepdish as dd
import pandas as pd

from .regression import ols_regression


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
            raw_data['scane_path_length'] = [raw_data['scane_path_length']]
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


def eye_features_analysis(config, features):
    # Check is dataframe is already there
    read_path = Path(__file__).parents[2] / config['eye_features_path']
    if read_path.is_file():
        eye_dataframe = pd.read_hdf(read_path, key='eye_dataframe')
    else:
        # Generate the dataframe
        eye_dataframe = _construct_eye_data(config, save_dataframe=True)

    # Select the features
    eye_dataframe = eye_dataframe[[
        'blinks', 'fixations', 'pupil_size', 'saccades', 'scane_path_length',
        'complexity', 'subject'
    ]]
    eye_subject_group = eye_dataframe.groupby(['subject', 'complexity']).mean()
    eye_subject_group.reset_index(inplace=True)

    models = []
    for feature in features:
        models.append(ols_regression('complexity', feature, eye_subject_group))
    return models, eye_subject_group
