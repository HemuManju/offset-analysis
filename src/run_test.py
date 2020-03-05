import yaml
from pathlib import Path

from tests.test_features import (test_fixations_map_coor,
                                 test_fixations_global_coor, test_user_actions)

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/tests/config.yml'
test_config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('run', 'Test fixations in map coor') as check, check():
    subject = 'T002'
    session = 'S002'
    test_fixations_map_coor(test_config, subject, session)

with skip_run('skip', 'Test fixations in global coor') as check, check():
    subject = 'T001'
    session = 'S003'
    test_fixations_global_coor(test_config, subject, session)

with skip_run('skip', 'Test user actions') as check, check():
    subject = 'T001'
    session = 'S002'
    test_user_actions(test_config, subject, session)
