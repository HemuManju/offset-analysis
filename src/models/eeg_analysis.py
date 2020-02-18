from pathlib import Path
import pandas as pd

import statsmodels.api as sm

from patsy import dmatrices


def eeg_features_analysis(config):
    # Load the dataframe
    read_path = Path(__file__).parents[2] / config['eeg_features_path']
    eeg_dataframe = pd.read_hdf(str(read_path), key='eeg_dataframe')

    # Select the features
    eeg_dataframe = eeg_dataframe[[
        'prob_distraction', 'prob_low_eng', 'prob_high_eng',
        'prob_ave_workload', 'complexity', 'subject'
    ]]

    eeg_subject_group = eeg_dataframe.groupby(['subject', 'complexity']).mean()
    eeg_subject_group.reset_index(inplace=True)

    y, X = dmatrices('prob_ave_workload ~ complexity',
                     data=eeg_subject_group,
                     return_type='dataframe')

    # Statistica analysis
    mod = sm.OLS(y, X)  # Describe model
    res = mod.fit()  # Fit model
    print(res.summary())


def eye_features_analysis(config):
    # Load the dataframe
    read_path = Path(__file__).parents[2] / config['eye_features_path']
    eye_dataframe = pd.read_hdf(str(read_path), key='eye_dataframe')

    eye_subject_group = eye_dataframe.groupby(['subject', 'complexity']).mean()

    eye_subject_group.reset_index(inplace=True)

    y, X = dmatrices('fixation ~ complexity',
                     data=eye_subject_group,
                     return_type='dataframe')

    # Statistica analysis
    mod = sm.OLS(y, X)  # Describe model
    res = mod.fit()  # Fit model
    print(res.summary())