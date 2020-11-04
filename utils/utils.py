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


def compute_action_mask(obs):
    # We want to prevent certain actions from being taken by appending a binary vector that
    # indicates which actions are possible
    stick_actions = obs["sticky_actions"]
    all_actions = np.ones(len(Action))
    all_actions[Action.Idle.value] = 0  # Turn off idle to force movement
    all_actions[Action.ReleaseDribble.value] = 0
    all_actions[Action.ReleaseSprint.value] = 0
    all_actions[Action.ReleaseDirection.value] = 0
    all_actions[Action.Slide.value] = 0

    if any(stick_actions[:8]):
        all_actions[Action.ReleaseDirection.value] = 1

    if stick_actions[8]:
        all_actions[Action.ReleaseSprint.value] = 1

    if stick_actions[9]:
        all_actions[Action.ReleaseDribble.value] = 1

    return all_actions