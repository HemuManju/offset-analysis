import yaml
from pathlib import Path

from data.extract_data import extract_offset_data
from data.clean_data import clean_eeg_data
from data.b_alert_data import write_mne_to_b_alert_edf
from data.utils import save_dataset, read_dataset

from features.offset_features import extract_offset_features

from models.eeg_analysis import eeg_features_analysis
from models.eye_analysis import (eye_features_analysis, calculate_fixations)
from models.game_analysis import (_get_user_actions, graph_with_user_actions,
                                  game_with_platoons)
from models.indv_analysis import individual_features_analysis

from visualization.visualize import (eeg_features_visualize, animate_bar_plot,
                                     eye_features_visualize,
                                     draw_fixation_in_map_coor,
                                     draw_fixation_in_global_coor,
                                     draw_platoon_in_map_coor)

from visualization.epoch_visualize import topo_visualize

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'Extract OFFSET data') as check, check():
    offset_data = extract_offset_data(config)
    save_path = Path(__file__).parents[1] / config['raw_offset_dataset']
    save_dataset(str(save_path), offset_data, save=True)

with skip_run('skip', 'Clean EEG data') as check, check():
    clean_dataset = clean_eeg_data(config['subjects'], config['sessions'],
                                   config)
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_dataset(str(save_path), clean_dataset, save=True)

with skip_run('skip', 'Save the EEG to B-alert') as check, check():
    write_mne_to_b_alert_edf(config, save_data=True)

with skip_run('skip', 'Save the clean EEG to B-alert') as check, check():
    write_mne_to_b_alert_edf(config, clean_with_ica=True, save_data=True)

with skip_run('skip', 'OFFSET feature extraction') as check, check():
    offset_features = extract_offset_features(config)
    save_path = Path(__file__).parents[1] / config['offset_features_path']
    save_dataset(str(save_path), offset_features, save=True)

with skip_run('skip', 'EEG feature analysis') as check, check():
    features = [
        'prob_distraction', 'prob_low_eng', 'prob_high_eng',
        'prob_ave_workload'
    ]
    models, eeg_subject_group = eeg_features_analysis(config, features)
    eeg_features_visualize(models, eeg_subject_group, features, 'complexity')

with skip_run('skip', 'Eye feature analysis') as check, check():
    features = ['n_fixations', 'n_saccades']
    models, eye_subject_group = eye_features_analysis(config, features)
    eye_subject_group.to_csv('Subject_Target_Info.csv')
    eye_features_visualize(models, eye_subject_group, features, 'complexity')

with skip_run('skip', 'Game feature analysis') as check, check():
    eye_features_analysis(config)

with skip_run('skip', 'EEG feature animation') as check, check():
    subject = config['subjects'][7]
    animate_bar_plot(config, subject, 0)

with skip_run('skip', 'EEG topoplot visualize') as check, check():
    read_path = Path(__file__).parents[1] / config['raw_eeg_dataset']
    data = read_dataset(str(read_path))
    epochs = data['sub-OFS_2008']['S005']['eeg'].load_data()
    topo_visualize(epochs, config)

with skip_run('skip', 'Eye fixation in map') as check, check():
    subject = config['subjects'][0]
    session = config['sessions'][0]
    fixations = calculate_fixations(config, subject, session, in_map=True)
    draw_fixation_in_map_coor(fixations, animate=False)

with skip_run('skip', 'Eye fixation in screen') as check, check():
    subject = config['subjects'][0]
    session = config['sessions'][0]
    fixations = calculate_fixations(config, subject, session)
    draw_fixation_in_global_coor(fixations)

with skip_run('skip', 'User actions') as check, check():
    subject = config['subjects'][0]
    session = config['sessions'][1]
    selected_nodes = _get_user_actions(config, subject, session)

with skip_run('skip', 'Visualize user actions') as check, check():
    subject = config['subjects'][8]
    session = config['sessions'][4]
    G = graph_with_user_actions(config, subject, session)

with skip_run('skip', 'Visualize platoons on the image') as check, check():
    subject = config['subjects'][8]
    session = config['sessions'][4]
    platoon_positions = game_with_platoons(config, subject, session)
    draw_platoon_in_map_coor(platoon_positions)

with skip_run('skip', 'Indiv difference analysis') as check, check():
    individual_features_analysis(config)
