import numpy as np
from rewards.distance_to_goal import dist_to_goal, dist_to_goal_reward, rule_based_reward
from utils.utils import sigmoid, angle_between_points, dist_between_points

values = {
    'prev_owner': 0
}


class EgoCentricObs(object):
    @staticmethod
    def parse(obs, action):
        active_index = obs['active']
        player_pos = obs['left_team'][active_index]
        player_vel = obs['left_team_direction'][active_index]
        player_tired_factor = obs['left_team_tired_factor'][active_index]
        player_role = obs['left_team_roles'][active_index]

        active_player = np.array([*player_pos, *player_vel, player_tired_factor, player_role])

        teammates = []
        for i in range(len(obs["left_team"])):
            # Add all your teammates
            if i == active_index:
                continue
            i_player_pos = obs['left_team'][i]
            i_player_vel = obs['left_team_direction'][i]
            i_player_tired_factor = obs['left_team_tired_factor'][i]
            i_player_role = obs['left_team_roles'][i]
            squashed_dist = sigmoid(dist_between_points(player_pos, i_player_pos))
            angle = angle_between_points(player_pos, i_player_pos)

            teammates.append([squashed_dist, np.cos(angle), np.sin(angle), i_player_vel[0], i_player_vel[1],
                              i_player_tired_factor, i_player_role])

        enemy_team = []
        for i in range(len(obs["right_team"])):
            i_player_pos = obs['right_team'][i]
            i_player_vel = obs['right_team_direction'][i]
            i_player_tired_factor = obs['right_team_tired_factor'][i]
            i_player_role = obs['right_team_roles'][i]
            squashed_dist = sigmoid(dist_between_points(player_pos, i_player_pos))
            angle = angle_between_points(player_pos, i_player_pos)

            enemy_team.append([squashed_dist, np.cos(angle), np.sin(angle), i_player_vel[0], i_player_vel[1],
                              i_player_tired_factor, i_player_role])
        teammates = np.array(teammates).flatten()
        enemy_team = np.array(enemy_team).flatten()
        curr_dist_to_goal = np.log(dist_to_goal(obs))  # Closer distance have larger variance, farther less important

        # get other information
        game_mode = [0 for _ in range(7)]
        game_mode[obs['game_mode']] = 1

        scalars = [*obs['ball'],
                   *obs['ball_direction'],
                   obs['ball_owned_team'],
                   obs['ball_owned_player'],
                   *obs['score'],
                   obs['steps_left'],
                   *game_mode,
                   *obs['sticky_actions'],
                   curr_dist_to_goal]

        lost_possession = False
        if obs["ball_owned_team"] != -1:
            lost_possession = values['prev_owner'] != obs["ball_owned_team"]
            values['prev_owner'] = obs["ball_owned_team"]

        scalars = np.r_[scalars].astype(np.float32)
        # get the actual scores and compute a reward
        l_score, r_score = obs['score']
        reward = rule_based_reward(obs, action, lost_possession)
        reward_info = l_score, r_score, reward
        # action_mask = compute_action_mask(obs).flatten()
        goal_pos = [1, 0]
        angle_to_goal = max(-np.pi / 2, min(np.pi / 2, angle_between_points(player_pos, goal_pos)))
        combined = np.concatenate([active_player.flatten(), teammates.flatten(),
                                   enemy_team.flatten(), scalars.flatten(), np.cos(angle_to_goal).flatten()])
        return combined, reward_info
