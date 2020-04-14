# from pathlib import Path
# import deepdish as dd

import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .game_features import _read_game_data

from .utils import findkeys, gamma_percentile


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


def _compute_epoch_info(config, subject, session):
    # Get the game
    game_state, game_time_stamps, time_kd_tree = _read_game_data(
        config, subject, session)
    pauses = list(findkeys(game_state, 'pause'))
    resumes = list(findkeys(game_state, 'resume'))

    pause_time, resume_time = [], []

    # TODO: Need to implement in a robust way
    # The last element is not pause
    for i, (pause, resume) in enumerate(zip(pauses, resumes)):
        if pause and not pauses[i - 1]:
            pause_time.append(game_time_stamps[i])
        if pause and not pauses[i + 1]:
            resume_time.append(game_time_stamps[i])
    assert len(pause_time) == len(resume_time), 'Length is different'

    # Find the difference
    epoch_lengths = [rt - pt for rt, pt in zip(resume_time, pause_time)]

    # Return epoch
    return epoch_lengths


def _compute_events(config, subject, session):
    raise NotImplementedError


def _get_action_type():
    raise NotImplementedError


def _sync_eeg_time(config, subject, session):
    epoch_length = _compute_epoch_info(config, subject, session)
    # print(epoch_length)

    # Create events according to game state

    # Get the EEG data and clean it with ICA

    # Epoched data

    return epoch_length


def _compute_time_threshold(config):
    time = []
    for subject in config['subjects']:
        for session in config['sessions']:
            time.append(_sync_eeg_time(config, subject, session))

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
