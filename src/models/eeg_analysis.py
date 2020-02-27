from pathlib import Path

import deepdish as dd
import pandas as pd

from .regression import ols_regression


def _construct_eeg_data(config, save_dataframe):
    eeg_dataframe = []
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
            raw_data = data['sub_OFS_' + subject][session]['eeg_features']
            temp_df = pd.DataFrame.from_dict(raw_data)

            # Add additional information
            temp_df['subject'] = subject
            temp_df['complexity'] = session_name[session]

            # Append it the the global dataframe
            eeg_dataframe.append(temp_df)

    eeg_dataframe = pd.concat(eeg_dataframe)

    if save_dataframe:
        save_path = Path(__file__).parents[2] / config['eeg_features_path']
        eeg_dataframe.to_hdf(save_path, key='eeg_dataframe')

    return eeg_dataframe


def eeg_features_analysis(config, features):
    # Check is dataframe is already there
    read_path = Path(__file__).parents[2] / config['eeg_features_path']
    if read_path.is_file():
        eeg_dataframe = pd.read_hdf(read_path, key='eeg_dataframe')
    else:
        # Generate the dataframe
        eeg_dataframe = _construct_eeg_data(config, save_dataframe=True)

    # Select the features
    eeg_dataframe = eeg_dataframe[[
        'prob_distraction', 'prob_low_eng', 'prob_high_eng',
        'prob_ave_workload', 'complexity', 'subject'
    ]]

    eeg_subject_group = eeg_dataframe.groupby(['subject', 'complexity']).mean()
    eeg_subject_group.reset_index(inplace=True)

    models = []
    for feature in features:
        models.append(ols_regression('complexity', feature, eeg_subject_group))
    return models, eeg_subject_group
