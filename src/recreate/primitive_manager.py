import numpy as np

from .primitives.planning.planners import SkeletonPlanning
from .utils import findkeys


class PrimitiveManager(object):
    def __init__(self, config, grid_map, action):
        """A base class to perform different primitives.

        Parameters
        ----------
        state_manager : instance
            An instance of state manager
        """
        self.config = config
        self.dt = self.config['simulation']['time_step']

        # Instance of primitives
        self.planning = SkeletonPlanning(config, grid_map)
        self.path_points = []
        self.patrol_poins = []
        self.action = action
        return None

    def check_engaging_distance(self, state):
        blue_cent_pos = list(findkeys(state, 'centroid_pos'))
        blue_cent_pos = np.array(blue_cent_pos, ndmin=2)
        dist = np.linalg.norm(self.action['centroid_pos'] - blue_cent_pos,
                              axis=1)
        idx = dist < self.config['experiment']['attack_distance']
        if idx.any():
            reset_centroid = blue_cent_pos[idx][0]
            self.action['centroid_pos'] = np.array(reset_centroid)
            engage = True
        else:
            engage = False

        # Check visibility also
        idx = dist < self.config['experiment']['detection_range']
        if idx.any():
            self.action['visibility'] = True
        return engage

    def execute_primitive(self, state, pause, update):
        """Perform primitive execution
        """
        primitives = {
            'formation': self.formation_primitive,
            'patrolling': self.patrolling_primitive,
        }
        # Check engaging distance
        engage = self.check_engaging_distance(state)

        # Execute the premitive
        if not pause and not engage:
            primitives[self.action['primitive']]()

        update_action = self.action.copy()
        return update_action

    def convert_pixel_ordinate(self, point, ispixel):
        """Convert the given point from pixel to cartesian co-ordinate or vice-versa.

        Parameters
        ----------
        point : list
            A list containing x and y position in pixel or cartesian space.
        ispixel : bool
            If True, the given input 'point' is in pixel space
            else it is in cartesian space.

        Returns
        -------
        list
            A converted point to pixel or cartesian space
        """
        if not ispixel:
            converted = [point[0] / 0.42871 + 145, point[1] / 0.42871 + 115]
        else:
            converted = [(point[0] - 145) * 0.42871,
                         (point[1] - 115) * 0.42871]
        return converted

    def get_spline_points(self):
        """Get the spline fit of path from start to end

        Returns
        -------
        list
            A list of points which are the fitted spline.
        """
        # Perform planning and fit a spline
        self.action['start_pos'] = self.action['centroid_pos']
        pixel_start = self.convert_pixel_ordinate(self.action['start_pos'],
                                                  ispixel=False)
        pixel_end = self.convert_pixel_ordinate(self.action['target_pos'],
                                                ispixel=False)
        path = self.planning.find_path(pixel_start, pixel_end, spline=False)

        # Convert to cartesian co-ordinates
        points = [
            self.convert_pixel_ordinate(point, ispixel=True) for point in path
        ]
        # As of now don't fit any splines
        if self.action['vehicles_type'] == 'uav':
            path_points = np.array(points[-1])
        else:
            path_points = np.array(points)
        return path_points

    def formation_primitive(self):
        """Performs formation primitive
        """
        # Do nothing as all the centroids not change

        return None

    def patrolling_primitive(self):
        """Perform patrolling primitive
        """
        if self.action['vehicles_type'] == 'ugv':
            # Initial formation
            if len(self.path_points) < 2:
                self.action['target_pos'] = self.action['sink_pos']
                self.path_points = self.get_spline_points()
                self.patrol_points = self.path_points.copy()
            else:
                distance = np.linalg.norm(self.action['centroid_pos'] -
                                          self.action['sink_pos'])

                if len(self.patrol_points) > 1 and distance > 2:
                    self.action['next_pos'] = self.patrol_points[0]
                    self.patrol_points = np.delete(self.patrol_points, 0, 0)
                else:
                    self.action['next_pos'] = self.action['sink_pos']
                    self.action['source_pos'], self.action[
                        'sink_pos'] = self.action['sink_pos'], self.action[
                            'source_pos']
                    self.patrol_points = np.flip(self.path_points, axis=0)

                # Update the centroid pos
                path = self.action['next_pos'] - self.action['centroid_pos']
                path_vel = (1 / self.dt) * path

                if np.linalg.norm(path_vel) > 0.43:
                    path_vel = (path_vel / np.linalg.norm(path_vel)) * 0.43

                self.action['centroid_pos'] = self.action[
                    'centroid_pos'] + path_vel * self.dt

        return None
