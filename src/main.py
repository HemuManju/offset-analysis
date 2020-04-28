import yaml
from pathlib import Path

from data.utils import save_dataset

from features.clean_eeg_data import clean_eeg_data
from features.game_features import extract_game_features

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'Clean eeg dataset') as check, check():
    cleaned_data = clean_eeg_data(config)
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_dataset(str(save_path), cleaned_data, save=True)

with skip_run('skip', 'Test option type') as check, check():
    for subject in ['2009', '2011', '2012', '2013', '2014', '2015', '2017']:
        for session in ['S002', 'S003']:
            print(subject, session)
            a = compute_option_type(config, subject, session)
            print(a.count('caution_option'), a.count('target_option'),
                  a.count('engage_option'))

with skip_run('run', 'Test option type') as check, check():
    subjects = ['2009', '2011', '2012', '2013', '2014', '2015', '2017']
    session = 'S003'
    extract_game_features(config, subjects[0], session)
