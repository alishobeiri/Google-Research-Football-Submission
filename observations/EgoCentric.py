import numpy as np
from rewards.distance_to_goal import dist_to_goal, dist_to_goal_reward, rule_based_reward
from rewards.possession_score_zero_sum import possession_score_reward
from utils.utils import sigmoid, angle_between_points, dist_between_points
from kaggle_environments.envs.football.helpers import *
from numpy import arctan2

goal_pos = [1.0, 0]

sticky_action_lookup = {val.name: i for i, val in enumerate(sticky_index_to_action)}


def dist_to_goal(obs):
    ball_pos = obs["ball"]
    dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(goal_pos))
    return dist


def dist_between_points(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def angle_between_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Need to scale all x values down
    angle = arctan2(float(dy), float(dx))
    return angle


def valid_team(team):
    return team == 'WeKick'


# Shoot, short pass, long pass, high pass
inter_action_vec_lookup = {Action.Shot.value: 0, Action.ShortPass.value: 1,
                           Action.LongPass.value: 2, Action.HighPass.value: 3}


class EgoCentricObs(object):
    def __init__(self):
        self.constant_lookup = dict(
            prev_team=-1,
            intermediate_action_vec=[0, 0, 0, 0],
            possession=False,
            prev_l_score=0,
            prev_r_score=0
        )

    def reset(self):
        self.constant_lookup = dict(
            prev_team=-1,
            intermediate_action_vec=[0, 0, 0, 0],
            possession=False,
            prev_l_score=0,
            prev_r_score=0
        )

    def action_mask(self, obs):
        # We want to prevent certain actions from being taken by appending a binary vector that
        # indicates which actions are possible
        stick_actions = obs["sticky_actions"]
        action_mask = np.ones(len(Action))

        action_mask[Action.ReleaseDribble.value] = 0
        action_mask[Action.ReleaseSprint.value] = 0
        action_mask[Action.ReleaseDirection.value] = 0

        if obs['ball_owned_team'] == -1:
            # Can move, sprint, idle
            action_mask[Action.LongPass.value] = 0
            action_mask[Action.HighPass.value] = 0
            action_mask[Action.ShortPass.value] = 0
            action_mask[Action.Shot.value] = 0
            action_mask[Action.Dribble.value] = 0
        elif obs['ball_owned_team'] == 0:
            # Can do everything but slide
            action_mask[Action.Slide.value] = 0
        elif obs['ball_owned_team'] == 1:
            action_mask[Action.LongPass.value] = 0
            action_mask[Action.HighPass.value] = 0
            action_mask[Action.ShortPass.value] = 0
            action_mask[Action.Shot.value] = 0
            action_mask[Action.Dribble.value] = 0

        # Handle sticky actions
        if any(stick_actions[:8]):
            action_mask[Action.ReleaseDirection.value] = 1

        if stick_actions[8]:
            action_mask[Action.ReleaseSprint.value] = 1

        if stick_actions[9]:
            action_mask[Action.ReleaseDribble.value] = 1

        return action_mask


    def parse(self, obs, prev_action=None):
        active_index = obs['active']
        player_pos = obs['left_team'][active_index]
        player_vel = obs['left_team_direction'][active_index]
        player_tired_factor = obs['left_team_tired_factor'][active_index]

        active_player = np.array([player_pos[0], player_pos[1] / 0.42,
                                  *player_vel, player_tired_factor])

        teammates = []
        for i in range(len(obs["left_team"])):
            # We purposely repeat ourselves to maintain consistency of roles
            i_player_pos = obs['left_team'][i]
            i_player_vel = obs['left_team_direction'][i]
            i_player_tired_factor = obs['left_team_tired_factor'][i]
            i_dist = dist_between_points(player_pos, i_player_pos)
            i_vel_mag = np.linalg.norm(i_dist)
            i_vel_ang = arctan2(i_player_vel[1], i_player_vel[0])
            angle = angle_between_points(player_pos, i_player_pos)
            teammates.append([i_player_pos[0], i_player_pos[1] / 0.42,
                              i_dist, np.cos(angle), np.sin(angle),
                              i_vel_mag, np.cos(i_vel_ang), np.sin(i_vel_ang),
                              i_player_tired_factor])

        enemy_team = []
        for i in range(len(obs["right_team"])):
            i_player_pos = obs['right_team'][i]
            i_player_vel = obs['right_team_direction'][i]
            i_player_tired_factor = obs['right_team_tired_factor'][i]
            i_dist = dist_between_points(player_pos, i_player_pos)
            i_vel_mag = np.linalg.norm(i_dist)
            i_vel_ang = arctan2(i_player_vel[1], i_player_vel[0])
            angle = angle_between_points(player_pos, i_player_pos)
            teammates.append([i_player_pos[0], i_player_pos[1] / 0.42,
                              i_dist, np.cos(angle), np.sin(angle),
                              i_vel_mag, np.cos(i_vel_ang), np.sin(i_vel_ang),
                              i_player_tired_factor])

        teammates = np.array(teammates).flatten()
        enemy_team = np.array(enemy_team).flatten()
        curr_dist_to_goal = dist_to_goal(obs)  # Closer distance have larger variance, farther less important

        # get other information
        game_mode = [0 for _ in range(7)]
        if (type(obs['game_mode']) is GameMode):
            game_mode[obs['game_mode'].value] = 1
        else:
            game_mode[obs['game_mode']] = 1

        sticky_action = [0 for _ in range(len(sticky_action_lookup))]
        if type(obs['sticky_actions']) is set:
            for action in obs['sticky_actions']:
                sticky_action[sticky_action_lookup[action.name]] = 1
        else:
            sticky_action = obs['sticky_actions']

        active_team = obs['ball_owned_team']
        prev_team = self.constant_lookup['prev_team']
        action_vec = self.constant_lookup['intermediate_action_vec']
        possession = False  # Determine if we have possession or not
        if ((active_team == 0 and prev_team == 0) or
                (active_team == 1 and prev_team == 0) or
                (active_team == 1 and prev_team == 1)):
            # Reset if lose the ball or keep the ball on pass
            self.constant_lookup['intermediate_action_vec'] = [0, 0, 0, 0]
            possession = False
        elif (active_team == -1 and prev_team == 0 and prev_action is not None):
            # Nobody owns right now and you had possession
            # Track prev actions
            if (type(prev_action) is Action and
                    prev_action.value in inter_action_vec_lookup):
                action_vec[inter_action_vec_lookup[prev_action.value]] = 1
            elif prev_action in inter_action_vec_lookup:
                action_vec[inter_action_vec_lookup[prev_action]] = 1
            possession = True

        if not possession and active_team == 0:
            possession = True

        if active_team != -1:
            self.constant_lookup['prev_team'] = active_team

        self.constant_lookup['possession'] = possession
        l_score, r_score = obs['score']
        prev_l_score, prev_r_score = self.constant_lookup['prev_l_score'], self.constant_lookup['prev_r_score']

        l_score_change = l_score - prev_l_score
        r_score_change = r_score - prev_r_score

        scalars = [obs['ball'][0],
                   obs['ball'][1] / 0.42,
                   *obs['ball_direction'],
                   obs['steps_left'] / 3000,
                   *game_mode,
                   curr_dist_to_goal,
                   *sticky_action,  # Tracks sticky actions
                   *action_vec,  # Tracks long term actions
                   l_score_change,
                   r_score_change,
                   possession,
                   active_team]

        scalars = np.r_[scalars].astype(np.float32)
        action_mask = self.action_mask(obs)
        combined = np.concatenate([active_player.flatten(), teammates.flatten(),
                                   enemy_team.flatten(), scalars.flatten(), action_mask.flatten()])
        done = False
        if obs['steps_left'] == 0:
            done = True

        reward = possession_score_reward(possession, l_score_change, r_score_change, l_score, r_score, done)

        self.constant_lookup['prev_r_score'] = r_score
        self.constant_lookup['prev_l_score'] = l_score

        return combined, (l_score, r_score, reward)