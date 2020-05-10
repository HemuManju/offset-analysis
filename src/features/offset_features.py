from pathlib import Path

import numpy as np
import pandas as pd
import deepdish as dd

from .eye_features import (extract_eye_features, extract_sync_eye_features)
from .eeg_features import (extract_b_alert_features, extract_sync_eeg_features)
from .game_features import extract_game_features, compute_option_type
from .indv_features import extract_individual_features
from .utils import nested_dict


def extract_offset_features(config):
    offset_features = {}
    for subject in config['subjects']:
        features = nested_dict()

        # Read individual difference
        features['individual_difference'] = extract_individual_features(
            config, subject)

        for session in config['sessions']:
            print(subject, session)

            # Read eye features
            features[session]['eye_features'] = extract_eye_features(
                config, subject, session)

            # Read eeg features
            features[session]['eeg_features'] = extract_b_alert_features(
                config, subject, session)

            # Read game features
            features[session]['game_features'] = extract_game_features(
                config, subject, session)

        offset_features['sub-OFS_' + subject] = features

    return offset_features


def extract_synched_features(config):
    offset_features = {}
    for subject in config['subjects']:
        features = nested_dict()

        for session in config['sessions']:
            print(subject, session)

            # Crop the eye data and time_stamps with respect to option time
            option_type, option_time = compute_option_type(
                config, subject, session)
            # Assert they are of same length
            assert len(option_type) == len(
                option_time), 'Data are of different length'

            # Extract eeg features
            features[session]['eeg_features'] = extract_sync_eeg_features(
                config, subject, session, option_type, option_time)

            # Extract eye features
            features[session]['eye_features'] = extract_sync_eye_features(
                config, subject, session, option_type, option_time)

        offset_features['sub-OFS_' + subject] = features

    return offset_features


def consolidate_eeg_features(features):
    if features:
        keys = features[0].keys()
    else:
        keys = [
            'cog_state', 'prob_ave_workload', 'prob_bds_workload',
            'prob_distraction', 'prob_fbds_workload', 'prob_high_eng',
            'prob_low_eng', 'prob_sleep_onset'
        ]
    df_eeg = pd.DataFrame(columns=keys)
    for i, feature in enumerate(features):
        df_temp = pd.DataFrame.from_dict(feature).mean()

        # Populate the dataframe
        df_eeg.loc[i] = df_temp.values
    return df_eeg


def consolidate_eye_features(features):
    df_eye = pd.DataFrame({
        'n_fixations': [],
        'n_saccades': [],
        'pupil_size': [],
        'scan_path_length': []
    })

    for i, feature in enumerate(features):
        # Populate the dataframe
        data = [
            feature['n_fixations'][0], feature['n_saccades'][0],
            np.nanmean(feature['pupil_size']), feature['scan_path_length'][0]
        ]
        df_eye.loc[i] = data

    return df_eye


def convert_eeg_eye_to_dataframe(config):
    options = ['target_option', 'engage_option', 'caution_option']

    eeg_eye_df = pd.DataFrame(np.empty((0, len(config['features']))),
                              columns=config['features'])

    for subject in config['subjects']:
        # Read the data of the subject
        read_path = Path(__file__).parents[2] / config['eeg_eye_features_path']
        read_group = '/sub-OFS_' + '/'.join([subject])
        data = dd.io.load(read_path, group=read_group)

        for session in config['sessions']:
            for option in options:
                # EEG features
                eeg_features = data[session]['eeg_features'][option]
                df_eeg = consolidate_eeg_features(eeg_features)

                # Eye features
                eye_features = data[session]['eye_features'][option]
                df_eye = consolidate_eye_features(eye_features)

                # Concatenate them to
                df_temp = pd.concat([df_eeg, df_eye], axis=1)

                # Add other details
                df_temp['option'] = option
                df_temp['complexity_type'] = session
                df_temp['subject'] = subject
                df_temp['option_type'] = config['option_type']

                eeg_eye_df = pd.concat([eeg_eye_df, df_temp],
                                       ignore_index=True)
    eeg_eye_df.drop(columns='engagement_index', inplace=True)
    return eeg_eye_df
