import yaml
from pathlib import Path

from data.extract_data import extract_eeg_data
from data.clean_data import clean_eeg_data
from data.mne_write_edf import write_mne_to_edf
from data.utils import save_dataset

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'Extract EEG data') as check, check():
    raw_eeg_data = extract_eeg_data(config)
    save_path = Path(__file__).parents[1] / config['raw_eeg_dataset']
    save_dataset(str(save_path), raw_eeg_data, save=True)

with skip_run('skip', 'Clean EEG data') as check, check():
    clean_dataset = clean_eeg_data(config['subjects'], config['sessions'],
                                   config)
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_dataset(str(save_path), clean_dataset, save=True)

with skip_run('run', 'Save the EEG to B-alert format') as check, check():
    write_mne_to_edf(config)
