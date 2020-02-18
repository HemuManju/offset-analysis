import matplotlib.pyplot as plt

import seaborn as sbs


def plot_settings():
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
