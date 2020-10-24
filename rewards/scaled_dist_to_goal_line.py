from rewards.distance_to_goal_line import dist_to_goal_line
import numpy as np

N_players = 11


def get_number_of_defenders_in_front(player, defenders):
    return sum([1 if player[0] > defender[0] else 0 for defender in defenders])


def scaled_dist_to_goal_line(obs, lost_possession=False):
    '''
    Scaled by number of defenders in front, the less defenders the more the reward
    the more defenders, the less the reward.

    Inverse reward helps to define reward to disincentivise random long balls for high reward
    '''
    player_pos = obs['left_team'][obs['active']]
    n_defenders_in_front = get_number_of_defenders_in_front(player_pos, obs["right_team"])

    dist = dist_to_goal_line(obs)
    scale = (N_players - n_defenders_in_front) / N_players
    inv_scale = n_defenders_in_front / N_players
    l_score, r_score = obs['score']

    if lost_possession:
        return inv_scale * (1 - dist**0.4)

    return scale * (1 - dist**0.4) + l_score
