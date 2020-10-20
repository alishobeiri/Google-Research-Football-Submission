import numpy as np

from rewards.distance_to_goal import dist_to_goal


class ObsParser(object):
    @staticmethod
    def parse(obs):
        # parse left players units
        l_units = [[x[0] for x in obs['left_team']], [x[1] for x in obs['left_team']],
                   [x[0] for x in obs['left_team_direction']], [x[1] for x in obs['left_team_direction']],
                   obs['left_team_tired_factor'], obs['left_team_yellow_card'],
                   obs['left_team_active'], obs['left_team_roles']
                   ]

        l_units = np.r_[l_units].T

        # parse right players units
        r_units = [[x[0] for x in obs['right_team']], [x[1] for x in obs['right_team']],
                   [x[0] for x in obs['right_team_direction']], [x[1] for x in obs['right_team_direction']],
                   obs['right_team_tired_factor'],
                   obs['right_team_yellow_card'],
                   obs['right_team_active'], obs['right_team_roles']
                   ]

        r_units = np.r_[r_units].T
        # combine left and right players units
        units = np.r_[l_units, r_units].astype(np.float32)

        # get other information
        game_mode = [0 for _ in range(7)]
        game_mode[obs['game_mode']] = 1
        scalars = [*obs['ball'],
                   *obs['ball_direction'],
                   *obs['ball_rotation'],
                   obs['ball_owned_team'],
                   obs['ball_owned_player'],
                   *obs['score'],
                   obs['steps_left'],
                   *game_mode,
                   *obs['sticky_actions']]

        scalars = np.r_[scalars].astype(np.float32)
        # get the actual scores and compute a reward
        l_score, r_score = obs['score']
        reward = dist_to_goal(obs)
        reward_info = l_score, r_score, reward
        combined = np.concatenate([units.flatten(), scalars.flatten()])
        return combined, reward_info
