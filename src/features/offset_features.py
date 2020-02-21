from pathlib import Path

import numpy as np
import pandas as pd
import deepdish as dd


def individual_features_with_time(config):
    session_name = {
        'S001': 'baseline',
        'S002': 'static_red',
        'S003': 'dynamic_red',
        'S004': 'static_red_smoke',
        'S005': 'dynamic_red_smoke'
    }

    column_names = ['mot', 'vs', 'completion_time', 'subject', 'complexity']
    individual_diff_df = pd.DataFrame(np.empty((0, len(column_names))),
                                      columns=column_names)

    # Read path
    read_path = Path(__file__).parents[2] / config['raw_offset_dataset']

    # Read the file
    df = []
    for subject in config['subjects']:
        individual_diff = dd.io.load(str(read_path),
                                     group='/sub_OFS_' + subject +
                                     '/individual_difference/')
        # Empty dataframe to store intermediate results

        for session in config['sessions']:
            group = '/sub_OFS_' + '/'.join(
                [subject, session, 'eeg', 'time_stamps'])
            eeg_time = dd.io.load(str(read_path), group=group)
            group = '/sub_OFS_' + '/'.join(
                [subject, session, 'eye', 'time_stamps'])
            eye_time = dd.io.load(str(read_path), group=group)
            group = '/sub_OFS_' + '/'.join(
                [subject, session, 'game', 'time_stamps'])
            game_time = dd.io.load(str(read_path), group=group)
            duration = eeg_time[-1] - eeg_time[0] + eye_time[-1] - eye_time[
                0] + game_time[-1] - game_time[0]
            data = [
                individual_diff['mot'], individual_diff['vs'], duration / 3,
                subject, session_name[session]
            ]
            df.append(data)

    individual_diff_df = pd.DataFrame(df, columns=column_names)

    # Clean before return
    individual_diff_df.dropna()
    return individual_diff_df
