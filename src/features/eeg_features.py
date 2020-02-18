import numpy as np
import pandas as pd


def extract_cognitive_features(config):
    session_name = {
        'S001': 'baseline',
        'S002': 'static_red',
        'S003': 'dynamic_red',
        'S004': 'static_red_smoke',
        'S005': 'dynamic_red_smoke'
    }

    column_names = [
        'session_num', 'elapsed _time', 'clock _time', 'prob_sleep_onset',
        'prob_distraction', 'prob_low_eng', 'prob_high_eng', 'cog_state',
        'prob_fbds_workload', 'prob_bds_workload', 'prob_ave_workload'
    ]
    eeg_data_frame = pd.DataFrame(np.empty((0, len(column_names))),
                                  columns=column_names)

    # Read the file
    for subject in config['subjects']:
        for session in config['sessions']:
            subject_file = 'sub_OFS_' + subject
            session_file = 'ses-' + session
            csv_file = ''.join([
                subject, '11000_ses-', session,
                '_task-T1_run-001.Classification.csv'
            ])
            read_path = ''.join([
                config['raw_xdf_path'], subject_file, '/', session_file,
                '/b-alert/', csv_file
            ])

            # Read the CSV file
            df = pd.read_csv(read_path)

            # Rename columns names
            df.columns = column_names

            # Add more information
            df['subject'] = subject
            df['complexity'] = session_name[session]
            eeg_data_frame = pd.concat([eeg_data_frame, df],
                                       ignore_index=True,
                                       sort=False)

    # Clean before return
    eeg_data_frame.dropna()
    eeg_data_frame = eeg_data_frame[~(eeg_data_frame == -99999).any(axis=1)]

    return eeg_data_frame
