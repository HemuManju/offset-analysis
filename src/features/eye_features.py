from pathlib import Path

import numpy as np
import pandas as pd

import deepdish as dd


def extract_eye_features(config):
    session_name = {
        'S001': 'baseline',
        'S002': 'static_red',
        'S003': 'dynamic_red',
        'S004': 'static_red_smoke',
        'S005': 'dynamic_red_smoke'
    }
    column_names = ['fixation']
    eye_data_frame = pd.DataFrame(np.empty((0, len(column_names))),
                                  columns=column_names)

    read_path = Path(__file__).parents[2] / config['raw_eye_dataset']
    all_data = dd.io.load(read_path)

    # Read the file
    for subject in ['2000', '2001', '2002', '2003', '2004', '2005']:
        for session in config['sessions']:
            eye_data = all_data['sub_OFS_' + subject][session]['eye']
            df = pd.DataFrame(eye_data, columns=column_names)
            # Add more information
            df['subject'] = subject
            df['complexity'] = session_name[session]
            eye_data_frame = pd.concat([eye_data_frame, df],
                                       ignore_index=True,
                                       sort=False)

    # Clean before return
    eye_data_frame.dropna()
    eye_data_frame = eye_data_frame[~(eye_data_frame == -99999).any(axis=1)]

    return eye_data_frame
