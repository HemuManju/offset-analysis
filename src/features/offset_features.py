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
