import numpy as np
goal_pos = [1, 0]
goal_post = [1, 0.04]


def dist_to_goal(obs):
    ball_pos = obs["ball"]
    dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(goal_pos))
    l_score, r_score = obs["score"]
    return (1 / dist)**0.4 + l_score
