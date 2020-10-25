import numpy as np
goal_pos = [1.04, 0]
goal_post = [1, 0.04]


def dist_to_goal(obs):
    ball_pos = obs["ball"]
    dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(goal_pos))
    return dist
