import matplotlib.pyplot as plt

import seaborn as sns


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
