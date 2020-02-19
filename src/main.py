import yaml
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from data.extract_data import (extract_eeg_data, extract_individual_diff,
                               extract_eye_data)
from data.clean_data import clean_eeg_data
from data.mne_write_edf import write_mne_to_edf
from data.utils import save_dataset, read_dataset

from features.eeg_features import extract_cognitive_features
from features.eye_features import extract_eye_features

from models.eeg_analysis import (eeg_features_analysis, eye_features_analysis)

from visualization.visualize import (eeg_features_visualize, animate_bar_plot,
                                     plot_settings, eye_features_visualize)
from visualization.epoch_visualize import topo_visualize

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('run', 'Extract EEG data') as check, check():
    raw_eeg_data = extract_eeg_data(config)
    save_path = Path(__file__).parents[1] / config['raw_eeg_dataset']
    save_dataset(str(save_path), raw_eeg_data, save=True)

with skip_run('skip', 'Clean EEG data') as check, check():
    clean_dataset = clean_eeg_data(config['subjects'], config['sessions'],
                                   config)
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_dataset(str(save_path), clean_dataset, save=True)

with skip_run('skip', 'Save the EEG to B-alert format') as check, check():
    write_mne_to_edf(config)

with skip_run('skip', 'Extract Individual difference') as check, check():
    individual_diff = extract_individual_diff(config)
    print(np.mean(individual_diff, axis=0))
    plot_settings()
    plt.scatter(individual_diff[:, 0], individual_diff[:, 1], s=100)
    plt.xlabel('Multi-object tracking task (% Correct)')
    plt.ylabel('Visual searching task (% Correct)')
    plt.tight_layout()
    plt.show()

with skip_run('skip', 'Extract Eye data') as check, check():
    raw_eye_data = extract_eye_data(config)
    save_path = Path(__file__).parents[1] / config['raw_eye_dataset']
    save_dataset(str(save_path), raw_eye_data, save=True)

with skip_run('skip', 'EEG feature extraction') as check, check():
    eeg_dataframe = extract_cognitive_features(config)
    save_path = Path(__file__).parents[1] / config['eeg_features_path']
    eeg_dataframe.to_hdf(str(save_path), key='eeg_dataframe')

with skip_run('skip', 'Eye feature extraction') as check, check():
    eye_dataframe = extract_eye_features(config)
    save_path = Path(__file__).parents[1] / config['eye_features_path']
    eye_dataframe.to_hdf(str(save_path), key='eye_dataframe')

with skip_run('skip', 'EEG feature analysis') as check, check():
    eeg_features_analysis(config)

with skip_run('skip', 'Eye feature analysis') as check, check():
    eye_features_analysis(config)

with skip_run('skip', 'EEG feature visualization') as check, check():
    eeg_features_visualize(config)

with skip_run('skip', 'Eye feature visualization') as check, check():
    eye_features_visualize(config)

with skip_run('skip', 'EEG feature animation') as check, check():
    subject = config['subjects'][7]
    animate_bar_plot(config, subject, 0)

with skip_run('skip', 'EEG topoplot visualize') as check, check():
    read_path = Path(__file__).parents[1] / config['raw_eeg_dataset']
    data = read_dataset(str(read_path))
    epochs = data['sub_OFS_2008']['S005']['eeg'].load_data()
    topo_visualize(epochs, config)
