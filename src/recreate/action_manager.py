import yaml
from pathlib import Path
import numpy as np

from .default_actions import red_team_actions
from .primitive_manager import PrimitiveManager


class ActionManager(object):
    def __init__(self, team_type):
        # Read the configuration file
        config_path = Path(
            __file__).parents[1] / 'recreate/config/simulation_config.yml'
        self.config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

        self.grid_map = np.load(self.config['map_save_path'] +
                                'occupancy_map.npy')
        default_actions = red_team_actions(self.config, team_type)
        self._init_platoons_setup(default_actions)
        return None

    def _init_platoons_setup(self, default_actions):
        """Initial setup of platoons with primitive execution class
        """
        self.uav_platoons = {}
        for i in range(self.config['simulation']['n_ugv_platoons']):
            vehicle_type = 'uav'
            key = vehicle_type + '_p_' + str(i + 1)
            action = default_actions[vehicle_type][key]
            self.uav_platoons[key] = PrimitiveManager(self.config,
                                                      self.grid_map, action)

        self.ugv_platoons = {}
        for i in range(self.config['simulation']['n_ugv_platoons']):
            vehicle_type = 'ugv'
            key = vehicle_type + '_p_' + str(i + 1)
            action = default_actions[vehicle_type][key]
            self.ugv_platoons[key] = PrimitiveManager(self.config,
                                                      self.grid_map, action)
        return None

    def check_shooting(self, state, complexity_state):
        raise NotImplementedError

    def recreate_states(self, states, game_states, updates):
        """Recreate the complexity states using the blue team states

        Parameters
        ----------
        states : list
            A list of blue team states
        game_states : list
            A list of game states e.g. pause or resume

        Returns
        -------
        complexity_states : list
            A list of complexity states reconstructed
        """
        complexity_states = []
        rate = 0
        for state, game_state, update in zip(states, game_states, updates):
            if update:  # rate % 2 == 0:
                collected_states = self.roll_actions(state,
                                                     game_state['pause'],
                                                     update)
            else:
                collected_states = complexity_states[-1]

            complexity_states.append(collected_states)
            rate += 1

        return complexity_states

    def roll_actions(self, state, game_state, update):
        """Roll the actions for all the UAV and UGV
        """
        complexity_states = {'uav': {}, 'ugv': {}}
        for key in self.uav_platoons:
            uav_platoon_state = self.uav_platoons[key].execute_primitive(
                state, game_state, update)
            complexity_states['uav'][key] = uav_platoon_state

        # Update all the ugv vehicles and write to parameter server
        for key in self.ugv_platoons:
            ugv_platoon_state = self.ugv_platoons[key].execute_primitive(
                state, game_state, update)
            complexity_states['ugv'][key] = ugv_platoon_state

        return complexity_states
