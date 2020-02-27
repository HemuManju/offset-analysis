import pandas as pd


def extract_b_alert_features(config, subject, session):
    column_names = [
        'session_num', 'elapsed _time', 'clock _time', 'prob_sleep_onset',
        'prob_distraction', 'prob_low_eng', 'prob_high_eng', 'cog_state',
        'prob_fbds_workload', 'prob_bds_workload', 'prob_ave_workload'
    ]

    # Read the file
    subject_file = 'sub_OFS_' + subject
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
    eeg_data = eeg_df.to_dict()
    return eeg_data
