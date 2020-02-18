import collections

import numpy as np
import pandas as pd

import mne
from .mne_import_xdf import read_raw_xdf


def read_xdf_eeg_data(config, subject, session):
    # Parameters
    subject_file = 'sub_OFS_' + subject
    session_file = 'ses-' + session
    xdf_file = 'sub_OFS_' + subject + '_ses-' + session + '_task-T1_run-001.xdf'

    # Read paths
    read_path = config[
        'raw_xdf_path'] + subject_file + '/' + session_file + '/' + xdf_file
    raw = read_raw_xdf(read_path)
    try:
        raw = raw.drop_channels(['ECG', 'AUX1', 'AUX2', 'AUX3'])
    except ValueError:
        pass

    raw.plot(block=True)
    # Filtering
    raw.notch_filter(60, filter_length='auto', phase='zero',
                     verbose=False)  # Line noise
    raw.filter(l_freq=0.5, h_freq=50, fir_design='firwin',
               verbose=False)  # Band pass filter

    # Channel information
    raw.set_montage(montage="standard_1020", set_dig=True, verbose=False)
    ch_info = {
        'Fp1': 'eeg',
        'F7': 'eeg',
        'F8': 'eeg',
        'T4': 'eeg',
        'T6': 'eeg',
        'T5': 'eeg',
        'T3': 'eeg',
        'Fp2': 'eeg',
        'O1': 'eeg',
        'P3': 'eeg',
        'Pz': 'eeg',
        'F3': 'eeg',
        'Fz': 'eeg',
        'F4': 'eeg',
        'C4': 'eeg',
        'P4': 'eeg',
        'POz': 'eeg',
        'C3': 'eeg',
        'Cz': 'eeg',
        'O2': 'eeg'
    }
    raw.set_channel_types(ch_info)

    # Epoching
    epoch_length = config['epoch_length']
    events = mne.make_fixed_length_events(raw, duration=epoch_length)
    epochs = mne.Epochs(raw,
                        events,
                        picks=['eeg'],
                        tmin=0,
                        tmax=config['epoch_length'],
                        baseline=(0, 0),
                        verbose=False)
    return epochs


def extract_eeg_data(config):
    subjects = config['subjects']
    sessions = config['sessions']
    data = {}
    eeg_data = collections.defaultdict(dict)

    for subject in subjects:
        for session in sessions:
            print(subject, session)
            epochs = read_xdf_eeg_data(config, subject, session)
            eeg_data[session]['eeg'] = epochs
        data['sub_OFS_' + subject] = eeg_data
    return data


def extract_individual_diff(config):
    individual_diff_data = np.zeros((len(config['subjects']), 2))

    for i, subject in enumerate(config['subjects']):
        for task in config['individual_diff']:
            # Save the file
            subject_file = 'sub_OFS_' + subject
            read_path = ''.join([
                config['raw_xdf_path'], subject_file, '/', task, '/', task,
                '_OFS_', subject, '.csv'
            ])
            if task == 'MOT':
                mot_df = pd.read_csv(read_path)
                individual_diff_data[i, 0] = np.mean(
                    mot_df['N_Reponse'].values) / 4
            else:
                vs_df = pd.read_csv(read_path)
                individual_diff_data[i,
                                     1] = np.sum(vs_df['Accuracy'].values) / 30
    return individual_diff_data


def read_eye_data(config, subject, session):
    # Parameters
    subject_file = 'sub_OFS_' + subject
    # Read paths
    read_path = ''.join([
        config['raw_xdf_path'], subject_file, '/tobii/', 'SUB_', subject,
        '_fixation_',
        session.lower(), '.csv'
    ])
    data = np.genfromtxt(read_path, delimiter=',')
    return data


def extract_eye_data(config):
    sessions = config['sessions']
    eye_data = {}
    for subject in ['2000', '2001', '2002', '2003', '2004', '2005']:
        data = collections.defaultdict(dict)
        for session in sessions:
            temp_data = read_eye_data(config, subject, session)
            data[session]['eye'] = temp_data
        eye_data['sub_OFS_' + subject] = data
    return eye_data
