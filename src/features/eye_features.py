from data.extract_data import read_xdf_eye_data

from .gaze.detectors import (blink_detection, fixation_detection,
                             saccade_detection)


def _compute_eye_features(epochs, time_stamps):
    time_stamps = (time_stamps - time_stamps[0]) * 1000
    channels = [
        'avg_pupil_dia', 'r_pixel_x', 'r_pixel_y', 'l_pixel_x', 'l_pixel_y'
    ]
    data = epochs.to_data_frame(picks=channels)
    data.fillna(0.0)

    # Verify the length of data
    assert data.shape[0] == len(time_stamps), "Data are not of same length"

    # Get x and y position
    pos_x = data[['r_pixel_x', 'l_pixel_x']].mean(axis=1).values
    pos_y = data[['r_pixel_y', 'l_pixel_y']].mean(axis=1).values

    # Get average of left and right eye
    _, blinks = blink_detection(pos_x, pos_y, time_stamps, minlen=5)
    _, fixations = fixation_detection(pos_x, pos_y, time_stamps, mindur=100)
    _, saccades = saccade_detection(pos_x.astype(int), pos_y.astype(int),
                                    time_stamps)

    return blinks, fixations, saccades


def extract_eye_features(config, subject, session):
    eye_data = {}

    # Read the eye data
    eye_epochs, time_stamps = read_xdf_eye_data(config, subject, session)

    # Calculate the features
    blinks, fixations, saccades = _compute_eye_features(
        eye_epochs, time_stamps)
    eye_data['blinks'] = blinks
    eye_data['fixations'] = fixations
    eye_data['saccades'] = saccades
    eye_data['time_stamps'] = time_stamps

    return eye_data
