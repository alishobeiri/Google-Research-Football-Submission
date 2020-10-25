import numpy as np
from rewards.distance_to_goal import dist_to_goal
from math import degrees
from numpy import arctan2

from rewards.scaled_dist_to_goal_line import get_number_of_defenders_in_front, scaled_dist_to_goal_line
from rewards.distance_to_goal_line import dist_to_goal_line

from kaggle_environments.envs.football.helpers import Action


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dist_between_points(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def angle_between_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Need to scale all x values down
    angle = degrees(arctan2(float(dy), float(dx)))
    return angle


def compute_action_mask(obs):
    # We want to prevent certain actions from being taken by appending a binary vector that
    # indicates which actions are possible
    stick_actions = obs["sticky_actions"]
    all_actions = np.ones(len(Action))
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

values = {
    'prev_owner': 0
}


class EgoCentricObs(object):
    @staticmethod
    def parse(obs):
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

        n_defenders_ahead = get_number_of_defenders_in_front(player_pos, obs["right_team"])
        curr_dist_to_goal = np.log(dist_to_goal(obs))  # Closer distance have larger variance, farther less important

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
                   *obs['sticky_actions'],
                   n_defenders_ahead,
                   curr_dist_to_goal]

        lost_possession = False
        if obs["ball_owned_team"] != -1:
            lost_possession = values['prev_owner'] != obs["ball_owned_team"]
            values['prev_owner'] = obs["ball_owned_team"]

        scalars = np.r_[scalars].astype(np.float32)
        # get the actual scores and compute a reward
        l_score, r_score = obs['score']
        reward = scaled_dist_to_goal_line(obs, lost_possession=lost_possession)
        reward_info = l_score, r_score, reward
        combined = np.concatenate([active_player.flatten(), teammates.flatten(),
                                   enemy_team.flatten(), scalars.flatten(), compute_action_mask(obs).flatten()])
        return combined, reward_info
