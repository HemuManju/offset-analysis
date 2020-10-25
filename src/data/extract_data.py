import pyxdf
import numpy as np
import pandas as pd
import ujson

from .mne_import_xdf import read_raw_xdf

from .utils import nested_dict


def read_xdf_eeg_data(config, subject, session):
    # Parameters
    subject_file = 'sub-OFS_' + subject
    session_file = 'ses-' + session
    xdf_file = ''.join(
        ['sub-OFS_', subject, '_ses-', session, '_task-T1_run-001.xdf'])

    # Read paths
    read_path = config[
        'raw_xdf_path'] + subject_file + '/' + session_file + '/' + xdf_file
    raw_eeg, time_info = read_raw_xdf(read_path)
    return raw_eeg, time_info


def read_xdf_eye_data(config, subject, session):
    # Parameters
    subject_file = 'sub-OFS_' + subject
    session_file = 'ses-' + session
    xdf_file = ''.join(
        ['sub-OFS_', subject, '_ses-', session, '_task-T1_run-001.xdf'])

    # Read paths
    read_path = config[
        'raw_xdf_path'] + subject_file + '/' + session_file + '/' + xdf_file
    raw_eye, time_info = read_raw_xdf(read_path, stream_id='Tobii_Eye_Tracker')
    return raw_eye, time_info


def read_individual_diff(config, subject):
    individual_diff_data = {}

    for task in config['individual_diff']:
        # Save the file
        subject_file = 'sub-OFS_' + subject
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
    subject_file = 'sub-OFS_' + subject
    session_file = 'ses-' + session
    xdf_file = ''.join(
        ['sub-OFS_', subject, '_ses-', session, '_task-T1_run-001.xdf'])

    # Read paths
    read_path = config[
        'raw_xdf_path'] + subject_file + '/' + session_file + '/' + xdf_file

    streams, fileheader = pyxdf.load_xdf(read_path)
    raw_game, time_info = None, None
    for stream in streams:
        if stream["info"]["name"][0] == 'parameter_server_states':
            raw_game = [ujson.loads(data[0]) for data in stream["time_series"]]
            time_info = stream
            break
    return raw_game, time_info


def extract_offset_data(config):
    subjects_data = {}

    for subject in config['subjects']:
        data = nested_dict()
        for session in config['sessions']:
            eye_data, eye_time_info = read_xdf_eye_data(
                config, subject, session)
            eeg_data, eeg_time_info = read_xdf_eeg_data(
                config, subject, session)
            game_data, game_time_info = read_xdf_game_data(
                config, subject, session)

            if config['extract_only_time']:
                data[session]['eye']['time_offsets'] = eye_time_info[
                    'time_offsets']
                data[session]['eeg']['time_offsets'] = eeg_time_info[
                    'time_offsets']
                data[session]['game']['time_offsets'] = game_time_info[
                    'time_offsets']
            else:
                # Save the data in dictionary
                data[session]['eye']['data'] = eye_data
                data[session]['eeg']['data'] = eeg_data
                data[session]['game']['data'] = game_data

            # Add time data
            data[session]['eye']['time_stamps'] = eye_time_info['time_stamps']
            data[session]['eeg']['time_stamps'] = eeg_time_info['time_stamps']
            data[session]['game']['time_stamps'] = game_time_info[
                'time_stamps']

        # Read individual difference
        indivdual_diff = read_individual_diff(config, subject)
        data['individual_difference'] = indivdual_diff

        # Store in the higher level dictionary
        subjects_data['sub-OFS_' + subject] = data
    return subjects_data
