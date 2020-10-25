import time
from pathlib import Path
from PIL import Image

import numpy
import deepdish as dd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pandas as pd
import seaborn as sns

from .graph_visualize import _network_legend

from features.gaze.gazeplotter import (draw_heatmap, draw_display,
                                       draw_eye_heatmap)


def _plot_settings():
    """Change the asthetics of the given figure (operators in place).

    Parameters
    ----------
    ax : matplotlib ax object

    """
    plt.style.use('clean')
    plt.rcParams.update({'font.family': "Arial"})
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rc('axes', axisbelow=True)
    return None


def convert_list_num(x):
    if isinstance(x, list):
        return x[0]
    else:
        return x


def _box_plots(dataframe, dependent, independent, axes):
    feature_dataframe = dataframe[[independent, dependent]]
    temp = feature_dataframe.dropna()

    temp[dependent] = temp.apply(lambda row: convert_list_num(row[dependent]),
                                 axis=1)
    sns.boxplot(x=independent, y=dependent, data=temp, ax=axes, width=0.45)
    plt.suptitle("")
    return None


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


def eeg_features_visualize(dataframe, features, independent):

    plt.style.use('clean_box')
    temp = dataframe[(dataframe['complexity'] == 'static_red') +
                     (dataframe['complexity'] == 'baseline')]

    title = ['High Engagement', 'Avg Mental Workload']
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[10, 4])
    for i, feature in enumerate(features):
        _box_plots(temp, feature, independent, ax[i])
        # ax.set_xticklabels([
        #     'Base line', 'Dynamic\n red team', 'Dynamic red\n team with smoke',
        #     'Static red\n team', 'Static red\n team with smoke'
        # ])
        ax[i].set_xticklabels(['Base line', 'Adversarial\n team'])
        ax[i].set_ylim([-0.1, 1.1])
        ax[i].set_ylabel('')
        ax[i].set_xlabel('Complexities')
        ax[i].set_title(title[i])
        ax[i].grid(True)

    ax[0].set_ylabel('Probability')
    plt.savefig('eeg-metrics.pdf', dpi=150)
    plt.show()
    return None


def eye_features_visualize(dataframe, features, independent):

    # Default plot settings
    plt.style.use('clean_box')
    temp = dataframe[(dataframe['complexity'] == 'static_red') +
                     (dataframe['complexity'] == 'baseline')]
    # title = ['Pupil Size', 'Fixations']
    labels = ['Avg Pupil Diameter (mm)', 'Num. Fixations']
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[12, 4])

    for i, feature in enumerate(features):
        _box_plots(temp, feature, independent, ax[i])
        # ax.set_xticklabels([
        #     'Base line', 'Dynamic\n red team', 'Dynamic red\n team with smoke',
        #     'Static red\n team', 'Static red\n team with smoke'
        # ])
        ax[i].set_xticklabels(['Base line', 'Adversarial\n team'])
        ax[i].set_ylabel(labels[i])
        ax[i].set_xlabel('Complexities')
        # ax[i].set_title(title[i])
        ax[i].grid(True)
    plt.savefig('eye-metrics.pdf', dpi=150)
    plt.show()
    return None


def animate_bar_plot(config, subject, session):
    session_name = {
        'S001': 'baseline',
        'S002': 'static_red',
        'S003': 'dynamic_red',
        'S004': 'static_red_smoke',
        'S005': 'dynamic_red_smoke'
    }

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
                                       session_name[session]]

    data = eeg_complexity_df[['prob_high_eng', 'prob_ave_workload']].values

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=[2.25, 10])
    colors = ['#6da04b', '#666666', '#e4e4e4', '#002f56']
    for i, value in enumerate(data):
        if i >= 23:
            ax[0].bar(['High\n Engagement'],
                      value[0],
                      color=colors[0],
                      edgecolor='k',
                      width=0.5)
            ax[0].set_ylim([0, 1.0])
            ax[0].grid(True)
            plt.tight_layout()
            plt.subplots_adjust(left=0.30, right=0.9, top=0.9, bottom=0.1)
            ax[0].set_ylabel('Probability')
            ax[1].bar(['Avg\n Mental Workload'],
                      value[1],
                      color=colors[1],
                      edgecolor='k',
                      width=0.5)
            plt.ylim([0, 1.0])
            plt.grid(True)
            plt.ylabel('Probability')
            plt.xlabel('Cognitive States')
            plt.pause(2)
            ax[0].cla()
            ax[1].cla()
            if i == 0:
                time.sleep(1)


def draw_fixation_in_map_coor(fixations, animate=True):
    read_path = Path(
        __file__).parents[1] / 'visualization/images/Benning_nodes.png'

    img = Image.open(read_path)
    img = img.resize((1500, 750))
    img = numpy.flip(numpy.array(img), axis=0)

    if animate:
        fig, ax = draw_display((1500, 750), imagefile=read_path)
        for i in range(0, len(fixations) - 5):
            fixation = fixations[i:i + 5]
            ax.imshow(img)
            draw_heatmap(fixation, dispsize=(1500, 750), ax=ax, imagefile=img)
            plt.cla()
    else:
        fig, ax = draw_display((1500, 750), imagefile=read_path)
        fixations = fixations[:-1]
        draw_heatmap(fixations, dispsize=(1500, 750), ax=ax, imagefile=img)
        plt.show()


def animate_eye_movements_in_map_coor(eye_positions, animate=True):

    read_path = Path(
        __file__).parents[1] / 'visualization/images/Benning_nodes.png'

    img = Image.open(read_path)
    img = img.resize((1500, 750))
    img = numpy.flip(numpy.array(img), axis=0)

    def update_plot(i, eye_positions, image_show):
        position = eye_positions[i:i + 10]
        image_data = draw_eye_heatmap(position, dispsize=(1500, 750))
        image_show.set_data(image_data * 0)
        image_show.set_data(image_data)
        return [image_show]

    if animate:
        fig, ax = draw_display((1500, 750), imagefile=read_path)
        ax = _network_legend(ax)
        init_data = draw_eye_heatmap(eye_positions[0:1], dispsize=(1500, 750))
        image_show = ax.imshow(init_data, cmap='jet', alpha=0.5)
        animation.FuncAnimation(fig,
                                update_plot,
                                frames=range(0,
                                             len(eye_positions) - 10, 10),
                                fargs=(eye_positions, image_show),
                                blit=True)
        plt.show()


def draw_eye_movements_in_map_coor(eye_positions, animate=True):

    read_path = Path(
        __file__).parents[1] / 'visualization/images/Benning_nodes.png'

    img = Image.open(read_path)
    img = img.resize((1500, 750))
    img = numpy.flip(numpy.array(img), axis=0)

    if animate:
        fig, ax = draw_display((1500, 750), imagefile=read_path)
        ax = _network_legend(ax)
        for i in range(0, len(eye_positions) - 5, 5):
            position = eye_positions[i:i + 5]
            ax.imshow(img)
            draw_eye_heatmap(position,
                             dispsize=(1500, 750),
                             ax=ax,
                             imagefile=img)
        plt.show()


def draw_platoon_in_map_coor(platoon_positions, animate=True):
    read_path = Path(
        __file__).parents[1] / 'visualization/images/Benning_nodes.png'

    img = Image.open(read_path)
    img = img.resize((1500, 750))
    img = numpy.array(img)
    color = ['#4A7DB3', '#4A7DB3', '#4A7DB3', '#519D3E', '#519D3E', '#519D3E']

    if animate:
        fig, ax = draw_display((1500, 750), imagefile=read_path)
        for i in range(0, len(platoon_positions) - 5, 5):
            position = platoon_positions[i, :, :]
            # Convert to pixel co-ordinates
            pixel_posx = position[:, 1] / 0.2 + 430
            pixel_posy = position[:, 0] / 0.2 + 340

            ax.imshow(img)
            ax.scatter(pixel_posx, pixel_posy, s=750, c=color, edgecolors='k')
            plt.pause(0.0001)
            plt.cla()
    return None


def draw_fixation_in_global_coor(fixations, animate=True):
    read_path = Path(__file__).parents[1] / 'visualization/images/Game.png'

    img = Image.open(read_path)
    img = img.resize((1500, 750))
    img = numpy.flip(numpy.array(img), axis=0)

    if animate:
        fig, ax = draw_display((1500, 750), imagefile=read_path)
        for i in range(0, len(fixations) - 5):
            fixation = fixations[i:i + 5]
            ax.imshow(img)
            draw_heatmap(fixation, dispsize=(1500, 750), ax=ax, imagefile=img)
            plt.cla()
    else:
        fig, ax = draw_display((1500, 750), imagefile=read_path)
        draw_heatmap(fixations, dispsize=(1500, 750), ax=ax, imagefile=img)
        plt.show()


def draw_user_action_fixations(fixations, user_action):
    read_path = Path(
        __file__).parents[1] / 'visualization/images/Benning_nodes.png'

    img = Image.open(read_path)
    img = img.resize((1500, 750))
    img = numpy.flip(numpy.array(img), axis=0)

    fig, ax = draw_display((1500, 750), imagefile=read_path)
    for i in range(0, len(fixations) - 5):
        fixation = fixations[i:i + 5]
        ax.imshow(img)
        draw_heatmap(fixation, dispsize=(1500, 750), ax=ax, imagefile=img)
        plt.cla()
