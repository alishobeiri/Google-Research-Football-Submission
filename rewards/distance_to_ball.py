import numpy as np
from kaggle_environments.envs.football.helpers import Action
from utils.utils import angle_between_points

goal_pos = [1.0, 0]
goal_post = [1, 0.04]


def friendly_player_dist_to_ball(obs):
    ball_pos = obs["ball"]
    active_index = obs['active']
    player_pos = obs['left_team'][active_index]

    dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(player_pos))
    return dist


def closest_defender_to_ball(obs):
    ball_pos = obs["ball"]
    closest_def_distance = np.min(np.linalg.norm(np.array(ball_pos[:2]) - np.array(obs['right_team'][1:]), axis=1))
    return closest_def_distance

