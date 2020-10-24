import numpy as np
from numpy import arccos, dot, pi, cross
from numpy.linalg import det, norm

goal_pos = np.array([1, 0])
goal_post_top = np.array([1, 0.04])
goal_post_bot = np.array([1, -0.04])


def distance_numpy(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
        return norm(P - A)
    if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
        return norm(P - B)
    return norm(cross(A-B, A-P))/norm(B-A)


def dist_to_goal_line(obs):
    ball_pos = obs["ball"]
    dist = distance_numpy(goal_post_top, goal_post_bot, np.array(ball_pos[:2]))
    return dist