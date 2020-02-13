import yaml
from pathlib import Path

import mne
import pyedflib
from data.create_data import create_eeg_data, read_xdf
from data.mne_write_edf import write_edf

from visualization.visualize import image_sequence

from utils import skip_run

config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'Create EEG data') as check, check():
    create_eeg_data(config)

with skip_run('skip', 'Run image sequence') as check, check():
    image_sequence(config)

with skip_run('run', 'B-alert data analysis') as check, check():
    data = read_xdf([], [], [])
    # data.plot(block=True)
    write_edf(data, 'test.edf', overwrite=True)
    new_data = mne.io.read_raw_edf(
        'data/raw/880310000.171116.115232.Signals.Raw.edf')

    write_edf(data, 'test.edf', overwrite=True)
    # print(new_data.info)
    # new_data.plot(block=True)

    data = mne.io.read_raw_edf('data/raw/test.edf')
    print(data.info)

with skip_run('skip', 'Test EDF reader') as check, check():
    data = pyedflib.EdfReader(
        'data/raw/880310000.171116.115232.Signals.Raw.edf')
