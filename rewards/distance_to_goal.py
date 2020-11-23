import numpy as np
from kaggle_environments.envs.football.helpers import Action
from utils.utils import angle_between_points

goal_pos = [1.0, 0]
goal_post = [1, 0.04]


def dist_to_goal(obs):
    ball_pos = obs["ball"]
    dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(goal_pos))
    return dist


def dist_to_goal_reward(obs):
    ball_pos = obs["ball"]
    dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(goal_pos))
    return 1 - dist**0.4


def rule_based_reward(obs, action=None, lost_possession=None):
    reward = 0
    if ((action == Action.HighPass) or
        (action == Action.LongPass) or
        (action == Action.ShortPass)):
        reward += 0.5
    elif action == Action.Shot:
        reward += 1.0

    l_score, r_score = obs['score']
    reward += l_score * 10  # If you score, you get a big big boost
    if lost_possession:
        reward += -10
    return reward
