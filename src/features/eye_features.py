import numpy
from data.extract_data import read_xdf_eye_data

from .gaze.detectors import (blink_detection, fixation_detection,
                             saccade_detection, scan_path)

from .utils import _time_kd_tree


def _convert_to_map_coor(eye_pos, map_pos):
    return [750 - map_pos[0] + eye_pos[0], 375 - map_pos[1] - eye_pos[1]]


def _compute_eye_features(epochs, time_stamps):
    milli_time_stamps = (time_stamps - time_stamps[0]) * 1000
    eye_data = {}

    channels = [
        'avg_pupil_dia', 'r_pixel_x', 'r_pixel_y', 'l_pixel_x', 'l_pixel_y'
    ]
    data = epochs.to_data_frame(picks=channels)
    data.fillna(0.0)

    # Verify the length of data
    assert data.shape[0] == len(
        milli_time_stamps), "Data are not of same length"

    # Get x and y position
    pos_x = data[['r_pixel_x', 'l_pixel_x']].mean(axis=1).values
    pos_y = data[['r_pixel_y', 'l_pixel_y']].mean(axis=1).values

    # Get the features
    _, blinks = blink_detection(pos_x, pos_y, milli_time_stamps, minlen=300)
    _, fixations = fixation_detection(pos_x,
                                      pos_y,
                                      milli_time_stamps,
                                      mindur=100)
    _, saccades = saccade_detection(pos_x.astype(int), pos_y.astype(int),
                                    milli_time_stamps)
    pupil_size = data['avg_pupil_dia'].values
    scan_path_length = scan_path(fixations)

    # Append the dictionary
    eye_data['blinks'] = blinks
    eye_data['fixations'] = fixations
    eye_data['n_fixations'] = [len(fixations)]
    eye_data['saccades'] = saccades
    eye_data['n_saccades'] = [len(saccades)]
    eye_data['pupil_size'] = pupil_size
    eye_data['scan_path_length'] = [scan_path_length]
    eye_data['time_stamps'] = time_stamps
    eye_data['time_kd_tree'] = _time_kd_tree(numpy.array(time_stamps, ndmin=2))

    # Convert eye data for map co-ordinate frame
    eye_data['pos'] = numpy.stack((pos_x, pos_y), axis=0).T
    return eye_data


def extract_eye_features(config, subject, session):

    # Read the eye data
    eye_epochs, time_stamps = read_xdf_eye_data(config, subject, session)

    # Calculate the features
    eye_data = _compute_eye_features(eye_epochs, time_stamps)
    return eye_data
