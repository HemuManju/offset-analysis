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

