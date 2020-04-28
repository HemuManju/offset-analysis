# from pathlib import Path
# import deepdish as dd

import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .game_features import _extract_action_info

from .utils import gamma_percentile


def extract_b_alert_features(config, subject, session):
    column_names = [
        'session_num', 'elapsed _time', 'clock _time', 'prob_sleep_onset',
        'prob_distraction', 'prob_low_eng', 'prob_high_eng', 'cog_state',
        'prob_fbds_workload', 'prob_bds_workload', 'prob_ave_workload'
    ]

    # Read the file
    subject_file = 'sub-OFS_' + subject
    session_file = 'ses-' + session
    csv_file = ''.join([
        subject, '11000_ses-', session, '_task-T1_run-001.Classification.csv'
    ])
    read_path = ''.join([
        config['raw_xdf_path'], subject_file, '/', session_file, '/b-alert/',
        csv_file
    ])

    # Read the CSV file to a dataframe
    eeg_df = pd.read_csv(read_path)

    # Rename columns names
    eeg_df.columns = column_names
    eeg_df.dropna()
    eeg_df = eeg_df[~(eeg_df == -99999).any(axis=1)]
    b_alert_features = eeg_df.to_dict()
    return b_alert_features


def _compute_engagement_level(epochs):
    raise NotImplementedError


def _compute_coherence(epochs):
    raise NotImplementedError


def _sync_eeg_time(config, subject, session):
    raise NotImplementedError


def _compute_time_threshold(config):
    time = []
    for subject in config['subjects']:
        for session in config['sessions']:
            _, _, epoch_lengths = _extract_action_info(config, subject,
                                                       session)
            time.append(epoch_lengths)

    time = numpy.array(sum(time, []))
    time = time[time <= 7.0]
    print(numpy.min(time), numpy.max(time))
    time_th = gamma_percentile(time, 0.99)
    print(time_th)
    sns.distplot(time)
    plt.show()
    return None


def extract_sync_eeg_features(config, subjects, sessions):
    for subject in subjects:
        for session in sessions:
            pass

            # Sync the eeg time with game time
