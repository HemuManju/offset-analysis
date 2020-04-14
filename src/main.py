import yaml
from pathlib import Path

from data.utils import save_dataset

from features.clean_eeg_data import clean_eeg_data

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'Clean eeg dataset') as check, check():
    cleaned_data = clean_eeg_data(config)
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_dataset(str(save_path), cleaned_data, save=True)

with skip_run('skip', 'Create eeg feature dataset') as check, check():
    cleaned_data = clean_eeg_data(config)
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_dataset(str(save_path), cleaned_data, save=True)