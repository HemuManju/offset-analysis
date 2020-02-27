import time

import deepdish as dd
import matplotlib.pyplot as plt

from pathlib import Path
import pandas as pd

import seaborn as sbs


def image_sequence(config):
    # Load the image files
    image_stream = dd.io.load(config['image_save_path'] + 'rgb_depth_seg.h5')

    # Figure
    fig, ax = plt.subplots(1, 3, figsize=[12, 5])
    titles = ['RGB Image', 'Depth Image', 'Segmented Image']
    plt.tight_layout()

    for stream in image_stream:
        for i in range(len(titles)):
            ax[i].imshow(stream[i], origin='lower')
            ax[i].title.set_text(titles[i])
        plt.pause(0.01)

        for i in range(len(titles)):
            ax[i].cla()


def _plot_settings():
    """Change the asthetics of the given figure (operators in place).

    Parameters
    ----------
    ax : matplotlib ax object

    """

    sbs.set_style("whitegrid")
    plt.rcParams.update({'font.family': "Arial"})
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['axes.labelweight'] = 'bold'

    return None


def _box_plots(models, dataframe, dependent, independent, axes):
    _plot_settings()
    feature_dataframe = dataframe[[independent, dependent]]
    feature_dataframe.boxplot(by=independent, ax=axes)
    plt.suptitle("")
    return None


def eeg_features_visualize(models, dataframe, features, independent):

    # Default plot settings
    _plot_settings()

    colors = ['#6da04b', '#666666', '#e4e4e4', '#002f56', '#2f9fd0']
    title = [
        'Distraction', 'Low Engagement', 'High Engagement',
        'Avg Mental Workload'
    ]

    for i, feature in enumerate(features):
        fig, ax = plt.subplots(figsize=[10, 5])
        _box_plots(models[i], dataframe, feature, independent, ax)

        ax.set_xticklabels([
            'Base line', 'Dynamic\n red team', 'Dynamic red\n team with smoke',
            'Static red\n team', 'Static red\n team with smoke'
        ])
        plt.ylabel('Probability')
        plt.xlabel('Complexities')
        plt.title(title[i])
        plt.tight_layout()
    plt.show()
    return None


def eye_features_visualize(models, dataframe, features, independent):

    # Default plot settings
    _plot_settings()

    colors = ['#6da04b', '#666666', '#e4e4e4', '#002f56', '#2f9fd0']
    title = ['Fixation', 'Sacaddes']

    for i, feature in enumerate(features):
        fig, ax = plt.subplots(figsize=[10, 5])
        _box_plots(models[i], dataframe, feature, independent, ax)

        ax.set_xticklabels([
            'Base line', 'Dynamic\n red team', 'Dynamic red\n team with smoke',
            'Static red\n team', 'Static red\n team with smoke'
        ])
        plt.ylabel('Probability')
        plt.xlabel('Complexities')
        plt.title(title[i])
        plt.tight_layout()
    plt.show()
    return None


def animate_bar_plot(config, subject, outcome):
    # Default plot settings
    _plot_settings()

    # Load the dataframe
    read_path = Path(__file__).parents[2] / config['eeg_features_path']
    eeg_dataframe = pd.read_hdf(str(read_path), key='eeg_dataframe')

    # Select the features
    eeg_dataframe = eeg_dataframe[[
        'prob_distraction', 'prob_low_eng', 'prob_high_eng',
        'prob_ave_workload', 'complexity', 'subject'
    ]]

    eeg_subject_df = eeg_dataframe[eeg_dataframe['subject'] == subject]
    eeg_complexity_df = eeg_subject_df[eeg_subject_df['complexity'] ==
                                       'dynamic_red_smoke']

    data = eeg_complexity_df[[
        'prob_distraction', 'prob_low_eng', 'prob_high_eng',
        'prob_ave_workload'
    ]].values

    # Plotting
    fig, ax = plt.subplots(figsize=[8, 5])
    colors = ['#6da04b', '#666666', '#e4e4e4', '#002f56']
    for i, value in enumerate(data):
        plt.bar([
            'Distraction', 'Low \n Engagement', 'High\n Engagement',
            'Avg\n Mental Workload'
        ],
                value,
                color=colors,
                edgecolor='k',
                width=0.5)
        plt.ylim([0, 1.0])
        plt.ylabel('Probability')
        plt.xlabel('Cognitive States')
        plt.pause(1)
        plt.tight_layout()
        plt.cla()
        if i == 0:
            time.sleep(20)
