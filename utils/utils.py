import numpy as np
from math import degrees
from numpy import arctan2

from kaggle_environments.envs.football.helpers import Action


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dist_between_points(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def angle_between_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Need to scale all x values down
    angle = arctan2(float(dy), float(dx))
    return angle


# def compute_action_mask(obs):
