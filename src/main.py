import yaml
from pathlib import Path

import deepdish as dd
from data.extract_data import extract_offset_data

from data.utils import save_dataset

from features.clean_eeg_data import clean_eeg_data
from features.game_features import compute_option_type
from features.offset_features import (extract_synched_features,
                                      convert_eeg_eye_to_dataframe)
from features.utils import read_data

from visualization.visualize import draw_fixation_in_global_coor
from visualization.epoch_visualize import topo_map
from visualization.regression_visualize import plot_mixed_effect_model
from visualization.time_visualize import plot_time_delays
from utils import skip_run, save_to_r_dataset

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'Clean eeg dataset') as check, check():
    cleaned_data = clean_eeg_data(config)
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_dataset(str(save_path), cleaned_data, save=False)

with skip_run('skip', 'Create eeg feature dataset') as check, check():
    cleaned_data = clean_eeg_data(config)
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_dataset(str(save_path), cleaned_data, save=True)

with skip_run('skip', 'Test options') as check, check():
    subjects = [
        '2009', '2011', '2012', '2013', '2014', '2015', '2017', '2018', '2020'
    ]
    for subject in subjects:
        compute_option_type(config, subject, 'S002')

with skip_run('skip', 'Extract epoched features') as check, check():
    synched_data = extract_synched_features(config)
    save_path = Path(__file__).parents[1] / config['eeg_eye_features_path']
    save_dataset(str(save_path), synched_data, save=True)

with skip_run('skip', 'Extract time data') as check, check():
    synched_data = extract_offset_data(config)
    save_path = Path(__file__).parents[1] / config['time_offset_dataset']
    save_dataset(str(save_path), synched_data, save=True)

with skip_run('skip', 'Convert features to dataframe') as check, check():
    eeg_eye_df = convert_eeg_eye_to_dataframe(config)
    save_path = Path(__file__).parents[1] / config['eeg_eye_dataframe_path']
    save_dataset(str(save_path), eeg_eye_df, save=True)

    # Convert to r dataframe and save
    read_path = Path(__file__).parents[1] / config['eeg_eye_dataframe_path']
    df = dd.io.load(read_path)
    save_path = Path(__file__).parents[1] / config['eeg_eye_r_dataset_path']
    save_to_r_dataset(df, str(save_path), save_as_csv=True)

with skip_run('skip', 'Draw fixation on global screen') as check, check():
    subject = config['subjects'][0]
    session = config['sessions'][0]
    eye_data = read_data(config, subject, session, 'eye_features')
    fixations = eye_data['fixations']
    draw_fixation_in_global_coor(fixations, animate=False)

with skip_run('skip', 'Draw topomaps') as check, check():
    subject = '2014'
    session = config['sessions'][0]
    topo_map(subject, config, session)

with skip_run('skip', 'Plot mixed effect model') as check, check():
    option = 'engage'
    id_difference = 'mot'
    plot_mixed_effect_model(config, option, id_difference)

with skip_run('run', 'Plot time delay') as check, check():
    plot_time_delays(config)
