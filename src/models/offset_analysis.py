from pathlib import Path
import pandas as pd

import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices

from visualization.utils import plot_settings


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


def individual_features_analysis(config):
    # Load the dataframe
    read_path = Path(__file__).parents[2] / config['individual_diff_path']
    individual_dataframe = pd.read_hdf(str(read_path),
                                       key='individual_dataframe')

    individual_subject_group = individual_dataframe[
        individual_dataframe['complexity'] == 'dynamic_red_smoke']

    individual_subject_group.reset_index(inplace=True)
    feature = 'vs'
    y, X = dmatrices('completion_time ~' + feature,
                     data=individual_subject_group,
                     return_type='dataframe')

    # Statistica analysis
    mod = sm.RLM(y / 60, X)  # Describe model
    resrlm = mod.fit()  # Fit model
    print(resrlm.summary())

    plot_settings()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(X[feature], y / 60, 'o')
    ax.plot(X[feature], resrlm.fittedvalues, 'g.-', label="RLM")
    ax.set_xlabel('Multi object tracking score (range 0-1)')
    ax.set_ylabel('Completion time (min)')
    ax.set_title('Dynamic red team with smoke')
    ax.grid(True)
    plt.show()
