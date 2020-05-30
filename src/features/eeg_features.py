import numpy as np
import pandas as pd
import seaborn as sns
import mne
from mne.connectivity import spectral_connectivity
from mne.time_frequency import psd_multitaper

import matplotlib.pyplot as plt

from data.extract_data import read_xdf_eeg_data

from .clean_eeg_data import _clean_with_ica, _filter_eeg
from .game_features import _extract_action_info

from .utils import (gamma_percentile, construct_time_kd_tree,
                    find_nearest_time_stamp)


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


def _compute_engagement_index(epochs, config):
    picks = mne.pick_types(epochs.info, eeg=True)
    ch_names = epochs.ch_names[picks[0]:picks[-1] + 1]
    psds, freqs = psd_multitaper(epochs,
                                 fmin=1.0,
                                 fmax=64.0,
                                 picks=picks,
                                 n_jobs=6,
                                 verbose=False,
                                 normalization='full')

    psd_band = []
    for freq_band in config['freq_bands']:
        psd_band.append(psds[:, :, (freqs >= freq_band[0]) &
                             (freqs < freq_band[1])].mean(axis=-1))
    # Form pandas dataframe
    data = np.concatenate(psd_band, axis=1)
    columns = [x + '_' + y for x in ch_names for y in config['band_names']]
    power_df = pd.DataFrame(data, columns=columns)

    # Engagement level; Feature Beta/(Alpha + Theta)
    num_bands = ['lower_Beta']
    num_electrodes = ch_names
    den_bands = ['Theta', 'total_Alpha']
    den_electrodes = ch_names
    numerator_features = [
        electrode + '_' + band for electrode in num_electrodes
        for band in num_bands
    ]
    denominator_features = [
        electrode + '_' + band for electrode in den_electrodes
        for band in den_bands
    ]
    alpha = [
        col for col in power_df[denominator_features].columns if 'Alpha' in col
    ]
    theta = [
        col for col in power_df[denominator_features].columns if 'Theta' in col
    ]
    beta_alpha_theta = power_df[numerator_features].mean(
        axis=1) / (power_df[alpha].mean(axis=1) + power_df[theta].mean(axis=1))
    return beta_alpha_theta.values


def _compute_coherence(cropped_eeg, config):
    epoch_length = config['epoch_length']
    events = mne.make_fixed_length_events(cropped_eeg, duration=epoch_length)
    epochs = mne.Epochs(cropped_eeg,
                        events,
                        picks=['eeg'],
                        tmin=0,
                        tmax=config['epoch_length'],
                        baseline=(0, 0),
                        verbose=False)
    fmin = (10.5, 22.5, 38)
    fmax = (12, 30, 42)
    # NOTE: Need to rewrite in a general way
    indices = (np.array([6, 6, 6]), np.array([8, 10, 12]))
    coh, freqs, _, _, _ = spectral_connectivity(epochs,
                                                fmin=fmin,
                                                fmax=fmax,
                                                indices=indices,
                                                faverage=True,
                                                verbose=False)
    keys = ['fz_pz_alpha', 'fz_t3_beta', 'fz_o1_gamma']
    values = np.diag(coh).tolist()
    coherence = dict(zip(keys, values))
    return coherence


def _compute_eeg_features(cropped_eeg, config):
    epoch_length = config['epoch_length']
    events = mne.make_fixed_length_events(cropped_eeg, duration=epoch_length)
    epochs = mne.Epochs(cropped_eeg,
                        events,
                        picks=['eeg'],
                        tmin=0,
                        tmax=config['epoch_length'],
                        baseline=(0, 0),
                        verbose=False)

    features = {}
    features['engagement_index'] = _compute_engagement_index(epochs, config)

    return features


def _compute_b_alert_features(config, subject, session, start_time):
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
    eeg_df = pd.read_csv(read_path,
                         skiprows=int(round(start_time)) - 1,
                         nrows=config['cropping_length'])
    # Rename columns names
    eeg_df.columns = column_names
    eeg_df.dropna()
    eeg_df = eeg_df[~(eeg_df == -99999).any(axis=1)]

    # Drop columns which are not necessary
    eeg_df.drop(columns=['session_num', 'elapsed _time', 'clock _time'],
                inplace=True)
    b_alert_features = eeg_df.to_dict(orient='list')

    return b_alert_features


def _compute_time_threshold(config):
    time = []
    for subject in config['subjects']:
        for session in config['sessions']:
            _, _, epoch_lengths = _extract_action_info(subject, session)
            time.append(epoch_lengths)

    time = np.array(sum(time, []))
    time = time[time <= 7.0]
    print(np.min(time), np.max(time))
    time_th = gamma_percentile(time, 0.99)
    print(time_th)
    sns.distplot(time)
    plt.show()
    return None


def extract_sync_eeg_features(config,
                              subject,
                              session,
                              option_type,
                              option_time,
                              use_balert=True):

    # Read subjects eye data
    eeg_data, time_stamps = read_xdf_eeg_data(config, subject, session)
    time_kd_tree = construct_time_kd_tree(np.array(time_stamps, ndmin=2).T)

    # Filter and clean the EEG data
    filtered_eeg = _filter_eeg(eeg_data, config)
    cleaned_eeg, ica = _clean_with_ica(filtered_eeg, config)

    eeg_features = {}
    eeg_features['target_option'] = []
    eeg_features['engage_option'] = []
    eeg_features['caution_option'] = []

    for option, time in zip(option_type, option_time):

        # Find the nearest time stamp
        nearest_time_stamp = find_nearest_time_stamp(time_kd_tree, time)

        # Get the start and end time of the epoch
        if config['option_type'] == 'pre_option':
            start_time = nearest_time_stamp['time'] - time_stamps[0] - config[
                'cropping_length']
            end_time = start_time + config['cropping_length']

        else:
            start_time = nearest_time_stamp['time'] - time_stamps[0]
            end_time = start_time + config['cropping_length']

        # Adjust the start time
        if start_time < 0:
            start_time = 0

        if not use_balert:
            # Copy the cleaned eeg
            temp_eeg = cleaned_eeg.copy()
            cropped_eeg = temp_eeg.crop(tmin=start_time, tmax=end_time)
            # Extract the features
            features = _compute_eeg_features(cropped_eeg, config)
            # Append the cleaned EEG data
            eeg_features['cleaned_eeg'] = cleaned_eeg
            eeg_features['ica'] = ica

        else:  # Extract B-alert features
            temp_eeg = cleaned_eeg.copy()
            cropped_eeg = temp_eeg.crop(tmin=start_time, tmax=end_time)
            coherence = _compute_coherence(cropped_eeg, config)
            features = _compute_b_alert_features(config, subject, session,
                                                 start_time)
            features.update(coherence)

        # Append the dictionary
        eeg_features[option].append(features)

    return eeg_features
