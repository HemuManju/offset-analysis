import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import plot_settings


def box_plots(models, dataframe, dependent, independent):
    plot_settings()
    feature_dataframe = dataframe[[independent, dependent]]
    feature_dataframe.boxplot(by=independent)
    return None


def plot_mixed_effect_model(config, option, id_difference):

    options = ['target', 'engage']
    labels = {'engage': 'Offensive', 'target': 'Target Search'}

    plt.style.use('clean')
    fig, ax = plt.subplots(figsize=(6, 4), nrows=1, ncols=2, sharey=True)
    for i, option in enumerate(options):
        data_path = config['r_regression_path']
        read_path = data_path + '_'.join(['df', 'vs', option]) + '.csv'
        data = pd.read_csv(read_path)
        data['subject'] = data['subject'].astype(str)
        data['option'] = data['option'].astype(str)
        sns.boxplot('response', 'fz_o1_gamma', data=data, width=0.5, ax=ax[i])
        ax[i].set_ylabel('')
        ax[i].set_xlabel('Tactics')
        ax[i].set_xticklabels(['Cautious', labels[option]])
    ax[0].set_ylabel('Fz-O1 Coherence (38-42 Hz)')
    plt.tight_layout(pad=0.1)
    plt.savefig('fz_o1_coherence.pdf')

    plt.show()
