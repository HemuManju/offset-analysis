import yaml
from pathlib import Path

from tests.test_features import test_fixations

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/tests/config.yml'
test_config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('run', 'Test fixations in map coor') as check, check():
    subject = 'T001'
    session = 'S001'
    test_fixations(test_config, subject, session)
