import numpy as np
from data.extract_data import read_xdf_eye_data

from .gaze.detectors import (blink_detection, fixation_detection,
                             saccade_detection, scan_path)
from .utils import (construct_time_kd_tree, find_nearest_time_stamp)


def _compute_eye_features(eye_data, time_stamps):
    milli_time_stamps = (time_stamps - time_stamps[0]) * 1000
    eye_features = {}

    channels = [
        'avg_pupil_dia', 'r_pixel_x', 'r_pixel_y', 'l_pixel_x', 'l_pixel_y'
    ]
    data = eye_data.to_data_frame(picks=channels)
    data.fillna(0.0)

    # Verify the length of data
    assert data.shape[0] == len(
        milli_time_stamps), "Data are not of same length"

    # Get x and y position
    pos_x = data[['r_pixel_x', 'l_pixel_x']].mean(axis=1).values
    pos_y = data[['r_pixel_y', 'l_pixel_y']].mean(axis=1).values

    # Get the features
    _, blinks = blink_detection(pos_x, pos_y, milli_time_stamps, minlen=10)
    _, fixations = fixation_detection(pos_x,
                                      pos_y,
                                      milli_time_stamps,
                                      mindur=50)
    _, saccades = saccade_detection(pos_x.astype(int), pos_y.astype(int),
                                    milli_time_stamps)
    pupil_size = data['avg_pupil_dia'].values
    scan_path_length = scan_path(fixations)

    # Append the dictionary
    eye_features['blinks'] = blinks
    eye_features['fixations'] = fixations
    eye_features['n_fixations'] = [len(fixations)]
    eye_features['saccades'] = saccades
    eye_features['n_saccades'] = [len(saccades)]
    eye_features['pupil_size'] = pupil_size
    eye_features['scan_path_length'] = [scan_path_length]
    eye_features['time_stamps'] = time_stamps
    eye_features['time_kd_tree'] = construct_time_kd_tree(
        np.array(time_stamps, ndmin=2).T)

    # Convert eye data for map co-ordinate frame
    eye_features['pos'] = np.stack((pos_x, pos_y), axis=0).T
    return eye_features


def extract_eye_features(config, subject, session):
    # Read the eye data
    eye_data, time_stamps = read_xdf_eye_data(config, subject, session)

    # Calculate the features
    eye_features = _compute_eye_features(eye_data, time_stamps)
    return eye_features


def extract_sync_eye_features(config, subject, session, option_type,
                              option_time):

    # Read subjects eye data
    eye_data, time_stamps = read_xdf_eye_data(config, subject, session)
    time_kd_tree = construct_time_kd_tree(np.array(time_stamps, ndmin=2).T)

    eye_features = {}
    eye_features['target_option'] = []
    eye_features['engage_option'] = []
    eye_features['caution_option'] = []

    for option, time in zip(option_type, option_time):
        # Copy the data
        temp_eye = eye_data.copy()

        # Find the nearest time stamp
        nearest_time_stamp = find_nearest_time_stamp(time_kd_tree, time)

        # Crop the data
        if config['option_type'] == 'pre_option':
            start_time = nearest_time_stamp['time'] - time_stamps[0] - config[
                'cropping_length']
            end_time = start_time + config['cropping_length']
        else:
            start_time = nearest_time_stamp['time'] - time_stamps[0]
            end_time = start_time + config['cropping_length']

        cropped_data = temp_eye.crop(tmin=start_time, tmax=end_time)

        # Extract the features
        features = _compute_eye_features(cropped_data, cropped_data.times)
        eye_features[option].append(features)

    return eye_features
