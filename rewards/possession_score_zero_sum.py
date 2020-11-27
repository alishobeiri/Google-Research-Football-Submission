from rewards.distance_to_ball import friendly_player_dist_to_ball, closest_defender_to_ball
from rewards.distance_to_goal import dist_to_goal, dist_to_goal_reward


def possession_score_reward(obs, possession, l_score_change, r_score_change, l_score, r_score, done):
    rew = 0

    ball_dist_to_goal = dist_to_goal(obs)
    ball_dist_to_goal = ball_dist_to_goal + 1e-10 if ball_dist_to_goal == 0 else ball_dist_to_goal

    ball_owned_team = obs['ball_owned_team']
    d_to_ball = friendly_player_dist_to_ball(obs)
    defender_dist_to_ball = closest_defender_to_ball(obs)
    if possession:
        if ball_owned_team == -1:
            # While ball is travelling don't give higher rewards
            rew += 0.02
        else:
            rew += 0.2 * (1/ball_dist_to_goal)
    else:
        rew -= 0.2 * ball_dist_to_goal * d_to_ball

    if l_score_change:
        rew += 5
    elif r_score_change:
        rew -= 5
    # print(rew)
    return rew