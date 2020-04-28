import numpy as np


class Shooting(object):
    def __init__(self):
        return None

    def shoot(self, n_team_blue, n_team_red, distance, type):
        if type == 'red':
            ratio = n_team_red / n_team_blue
        else:
            ratio = n_team_blue / n_team_red
        p = np.random.gamma(shape=ratio, scale=0.75)
        return p

    def shoot_original(self, vehicles, n_team_red, n_team_blue, distance,
                       type):
        if type == 'red':
            ratio = n_team_red / n_team_blue
        else:
            ratio = n_team_blue / n_team_red

        for vehicle in vehicles:
            vehicle.ammo -= 1
            p_req = 1 - (distance - min_dist_to_shoot) / (distance -
                                                          min_dist_to_kill)
            if vehicle.ammo < 0 or np.random.rand() < p_req:
                vehicle.functional = False

        return vehicles
