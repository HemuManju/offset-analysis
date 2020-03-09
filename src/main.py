import yaml
from pathlib import Path

from data.create_data import create_eeg_data

from utils import skip_run

config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('run', 'Create EEG data') as check, check():
    create_eeg_data(config)

with skip_run('run', 'Create EEG data') as check, check():
    create_eeg_data(config)