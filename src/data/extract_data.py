import pyxdf
import numpy as np
import pandas as pd

import ujson

from .mne_import_xdf import read_raw_xdf

from .utils import nested_dict


def read_xdf_eeg_data(config, subject, session):
    # Parameters
    subject_file = 'sub_OFS_' + subject
    session_file = 'ses-' + session
    xdf_file = ''.join(
        ['sub_OFS_', subject, '_ses-', session, '_task-T1_run-001.xdf'])

    # Read paths
    read_path = config[
        'raw_xdf_path'] + subject_file + '/' + session_file + '/' + xdf_file
    raw_eeg, time_stamps = read_raw_xdf(read_path)
    return raw_eeg, time_stamps


def read_xdf_eye_data(config, subject, session):
    # Parameters
    subject_file = 'sub_OFS_' + subject
    session_file = 'ses-' + session
    xdf_file = ''.join(
        ['sub_OFS_', subject, '_ses-', session, '_task-T1_run-001.xdf'])

    # Read paths
    read_path = config[
        'raw_xdf_path'] + subject_file + '/' + session_file + '/' + xdf_file
    raw_eye, time_stamps = read_raw_xdf(read_path,
                                        stream_id='Tobii_Eye_Tracker')
    return raw_eye, time_stamps


def read_individual_diff(config, subject):
    individual_diff_data = {}

    for task in config['individual_diff']:
        # Save the file
        subject_file = 'sub_OFS_' + subject
        read_path = ''.join([
            config['raw_xdf_path'], subject_file, '/', task, '/', task,
            '_OFS_', subject, '.csv'
        ])
        if task == 'MOT':
            mot_df = pd.read_csv(read_path)
            individual_diff_data['mot'] = np.mean(
                mot_df['N_Reponse'].values) / 4
        else:
            vs_df = pd.read_csv(read_path)
            individual_diff_data['vs'] = np.sum(vs_df['Accuracy'].values) / 30
    return individual_diff_data


def read_xdf_game_data(config, subject, session):
    # Parameters
    subject_file = 'sub_OFS_' + subject
    session_file = 'ses-' + session
    xdf_file = ''.join(
        ['sub_OFS_', subject, '_ses-', session, '_task-T1_run-001.xdf'])

    # Read paths
    read_path = config[
        'raw_xdf_path'] + subject_file + '/' + session_file + '/' + xdf_file

    streams, fileheader = pyxdf.load_xdf(read_path)
    for stream in streams:
        raw_game = []
        if stream["info"]["name"][0] == 'parameter_server_states':
            for data in stream["time_series"]:
                raw_game.append(ujson.loads(data[0]))
            time_stamps = stream["time_stamps"]
            break

    return raw_game, time_stamps


def extract_offset_data(config):
    sessions = config['sessions']
    subjects_data = {}

    for subject in config['subjects']:
        data = nested_dict()
        for session in sessions:
            eye_data, eye_time_stamp = read_xdf_eye_data(
                config, subject, session)
            eeg_data, eeg_time_stamp = read_xdf_eeg_data(
                config, subject, session)
            game_data, game_time_stamp = read_xdf_game_data(
                config, subject, session)

            # Save the data in dictionary
            data[session]['eye']['data'] = eye_data
            data[session]['eye']['time_stamps'] = eye_time_stamp
            data[session]['eeg']['data'] = eeg_data
            data[session]['eeg']['time_stamps'] = eeg_time_stamp
            data[session]['game']['data'] = game_data
            data[session]['game']['time_stamps'] = game_time_stamp

        # Read individual difference
        indivdual_diff = read_individual_diff(config, subject)
        data['individual_difference'] = indivdual_diff

        # Store in the higher level dictionary
        subjects_data['sub_OFS_' + subject] = data
    return subjects_data
