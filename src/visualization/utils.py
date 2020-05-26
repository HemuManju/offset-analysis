import matplotlib.pyplot as plt

import seaborn as sns


def _convert_to_global_coor(fixations, window_size):
    width = int(window_size[0] / 2)
    height = int(window_size[1] / 2)
    for fixation in fixations:
        fixation[3] = width + fixation[3]
        fixation[4] = height + fixation[4]
    return fixations


def plot_settings():
    """Change the asthetics of the given figure (operators in place).

    Parameters
    ----------
    ax : matplotlib ax object

    """
    plt.rcParams.update({'font.family': "Arial"})
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rc('axes', axisbelow=True)
    sns.axes_style("ticks")

    return None
