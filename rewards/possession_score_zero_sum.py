from rewards.distance_to_ball import friendly_player_dist_to_ball, closest_defender_to_ball
from rewards.distance_to_goal import dist_to_goal, dist_to_goal_reward, dist_to_goal_y, dist_to_goal_x
from kaggle_environments.envs.football.helpers import *


def possession_score_reward(obs, possession, l_score_change, r_score_change, action, l_score, r_score, done):
    rew = 0

    ball_dist_to_goal = dist_to_goal(obs)

    ball_dist_to_goal_x = dist_to_goal_x(obs)
    ball_dist_to_goal_y = dist_to_goal_y(obs)

    ball_dist_to_goal_x = ball_dist_to_goal_x + 1e-10 if ball_dist_to_goal_x == 0 else ball_dist_to_goal_x
    ball_dist_to_goal_y = ball_dist_to_goal_y + 1e-10 if ball_dist_to_goal_y == 0 else ball_dist_to_goal_y

    ball_owned_team = obs['ball_owned_team']
    d_to_ball = friendly_player_dist_to_ball(obs)
    if possession:
        if ball_owned_team == -1:
            # While ball is travelling don't give higher rewards
            rew += 0.01
        else:
            rew += 0.02 * (1 - ball_dist_to_goal_x / 2) * (1 - ball_dist_to_goal_y / 0.42)
    else:
        rew -= 0.4 * (ball_dist_to_goal_x/2) * (ball_dist_to_goal_y/0.42) * d_to_ball

    if l_score_change:
        rew += 20
    elif r_score_change:
        rew -= 20

    return rew