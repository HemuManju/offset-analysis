from models.eye_analysis import fixation_in_map_coor

from visualization.visualize import draw_fixation_in_map_coor


def test_fixations(config, subject, session):
    fixations_map_coor = fixation_in_map_coor(config, subject, session)
    draw_fixation_in_map_coor(fixations_map_coor)
    return None
