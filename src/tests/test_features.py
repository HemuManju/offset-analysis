from models.eye_analysis import calculate_fixations
from models.game_analysis import get_user_actions

from visualization.visualize import (draw_fixation_in_map_coor,
                                     draw_fixation_in_global_coor)


def test_fixations_map_coor(config, subject, session):
    fixations = calculate_fixations(config, subject, session, in_map=True)
    draw_fixation_in_map_coor(fixations)
    return None


def test_fixations_global_coor(config, subject, session):
    fixations = calculate_fixations(config, subject, session)
    draw_fixation_in_global_coor(fixations)
    return None


def test_user_actions(config, subject, session):
    selected_nodes = get_user_actions(config, subject, session)
    print(selected_nodes)
    return None