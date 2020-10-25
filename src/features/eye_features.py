from pathlib import Path
import numpy
from data.extract_data import read_xdf_eye_data

import pandas as pd

from .gaze.detectors import (blink_detection, fixation_detection,
                             saccade_detection, scan_path)

from .utils import _time_kd_tree


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
                                      mindur=50)
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
    eye_data['time_kd_tree'] = _time_kd_tree(
        numpy.array(time_stamps, ndmin=2).T)

    # Convert eye data for map co-ordinate frame
    eye_data['pos'] = numpy.stack((pos_x, pos_y), axis=0).T
    return eye_data


def extract_eye_features(config, subject, session):
    # Read the eye data
    eye_epochs, time_stamps = read_xdf_eye_data(config, subject, session)

    # Calculate the features
    eye_data = _compute_eye_features(eye_epochs, time_stamps)
    return eye_data


def _without_keys(dictionary, exclude):
    for i in exclude:
        dictionary.pop(i)
    return dictionary


def extract_eye_dataframe(config):
    eye_dataframe = []
    session_name = {
        'S001': 'baseline',
        'S002': 'static_red',
        'S003': 'dynamic_red',
        'S004': 'static_red_smoke',
        'S005': 'dynamic_red_smoke'
    }

    for subject in config['subjects']:
        for session in config['sessions']:
            raw_data = extract_eye_features(config, subject, session)
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

    save_path = Path(__file__).parents[2] / config['eye_features_path']
    eye_dataframe.to_hdf(save_path, key='eye_dataframe')

    return eye_dataframe
