from .eye_features import extract_eye_features
from .eeg_features import extract_b_alert_features
from .game_features import extract_game_features
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
            # Read eye features
            features[session]['eye_features'] = extract_eye_features(
                config, subject, session)

            # Read eeg features
            features[session]['eeg_features'] = extract_b_alert_features(
                config, subject, session)

            # Read game features
            features[session]['game_features'] = extract_game_features(
                config, subject, session)

        offset_features['sub_OFS_' + subject] = features

    return offset_features
