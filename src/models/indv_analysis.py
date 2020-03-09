from pathlib import Path
import pandas as pd

import deepdish as dd
import numpy

import statsmodels.api as sm
# import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from patsy import dmatrices

from visualization.utils import plot_settings


def _construct_individual_diff_data(config, save_dataframe):
    indv_dataframe = []
    session_name = {
        'S001': 'baseline',
        'S002': 'static_red',
        'S003': 'dynamic_red',
        'S004': 'static_red_smoke',
        'S005': 'dynamic_red_smoke'
    }

    read_path = Path(__file__).parents[2] / config['offset_features_path']
    data = dd.io.load(read_path)

    for subject in config['subjects']:
        for session in config['sessions']:
            raw_indv_data = data['sub-OFS_' + subject]['individual_difference']
            raw_game_data = data['sub-OFS_' +
                                 subject][session]['game_features']
            # Calculate the distance
            nodes_selected = raw_game_data['selected_node_pos'][-1][0]
            distance = numpy.linalg.norm(nodes_selected)

            # Calculate the completion time
            time = raw_game_data['time_stamps']
            completion_time = time[-1] - time[0]

            # Casualities
            casualities = numpy.sum(raw_game_data['casualities'])
            if casualities == 0:
                casualities = 1

            # Need to orient and transpose,
            # because the data are of not same length
            temp_df = pd.DataFrame.from_dict(raw_indv_data,
                                             orient='index').transpose()
            # Add additional information
            temp_df['distance'] = distance
            temp_df['completion_time'] = completion_time
            temp_df['casualities'] = casualities
            temp_df['subject'] = subject
            temp_df['complexity'] = session_name[session]

            # Append it the the global dataframe
            indv_dataframe.append(temp_df)

    indv_dataframe = pd.concat(indv_dataframe)

    if save_dataframe:
        save_path = Path(__file__).parents[2] / config['individual_diff_path']
        indv_dataframe.to_hdf(save_path, key='indv_dataframe')

    return indv_dataframe


def individual_features_analysis(config):
    # Check is dataframe is already there
    read_path = Path(__file__).parents[2] / config['individual_diff_path']
    if read_path.is_file():
        individual_dataframe = pd.read_hdf(read_path, key='indv_dataframe')
    else:
        # Generate the dataframe
        individual_dataframe = _construct_individual_diff_data(
            config, save_dataframe=True)

    individual_dataframe = individual_dataframe[
        individual_dataframe['complexity'] == 'baseline']

    individual_subject_group = individual_dataframe.groupby(['subject']).mean()

    individual_subject_group.reset_index(inplace=True)
    individual_subject_group['velocity'] = individual_subject_group[
        'completion_time'] / individual_subject_group['distance']
    individual_subject_group = individual_subject_group[
        individual_subject_group['subject'] != '2017']

    feature = 'vs'

    # Statistica analysis
    y, X = dmatrices('velocity ~' + feature,
                     data=individual_subject_group,
                     return_type='dataframe')
    mod = sm.RLM(y, X)  # Describe model
    resrlm = mod.fit()  # Fit model
    print(resrlm.summary())

    plot_settings()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(individual_subject_group[feature],
            individual_subject_group['velocity'], 'o')
    ax.plot(individual_subject_group[feature],
            resrlm.fittedvalues,
            'g.-',
            label="RLM")
    ax.set_xlabel('Visual search score (range 0-1)')
    ax.set_ylabel('Normalised Completion time (s/m)')
    ax.set_title('Baseline Task')
    ax.grid(True)
    plt.tight_layout(pad=0)
    plt.savefig('vs_completion_time.pdf')
    plt.show()
