from kaggle_environments.envs.football.helpers import *
import math
import numpy as np


@human_readable_agent
def agent(obs):

    def distance(player_one, player_two):
        # Compute the distance between two players
        return math.hypot(player_one[0] - player_two[0], player_one[1] - player_two[1])

    def angle(coord_one, coord_two):
        # Compute the angle between two coordinates
        return math.degrees(math.atan2(
            coord_two[1] - coord_one[1], coord_two[0] - coord_one[0]))

    def closest_player(controlled_player_position, team='left_team'):
        # Find the index of the player closest to the player in posession
        # Use team = 'right_team' to get the closest opponent
        distances = obs[team].copy()
        for i, teammate_position in enumerate(obs[team]):
            distances[i] = distance(
                controlled_player_position, teammate_position)
        return np.argmin(distances)

    def coordinate_to_direction(coord_one, coord_two):
        # Given coordinate between a player and anything (player, goal ...), output cardinal direction
        deg = angle(coord_one, coord_two)
        if -22.5 <= deg and deg <= 22.5:
            return Action.Right
        elif 22.5 < deg and deg <= 67.5:
            return Action.TopRight
        elif 67.5 < deg and deg <= 112.5:
            return Action.Top
        elif 112.5 <= deg and deg <= 157.5:
            return Action.TopLeft
        elif deg > 157.5 or deg <= -157.5:
            return Action.Left
        elif deg <= -112.5:
            return Action.BottomLeft
        elif deg <= -67.5:
            return Action.Bottom
        elif deg <= -22.5:
            return Action.BottomRight

    def direction_check(action, direction):
        # Sticky is a N hot encoded array for whether the player is performing an action
        # Here, we check that the player is directed properly or we let it perform an action
        if direction in obs['sticky_actions'][0:9]:
            return action
        else:
            return direction

    def expected_goal(player):
        # Compute expected goals from paper :https://www.researchgate.net/publication/240641737_Estimating_the_probability_of_a_shot_resulting_in_a_goal_The_effects_of_distance_angle_and_space
        # Football rules: size of goal : 8 yard by 8 feet
        def to_yard(coord): return [coord[0] * 4 / 0.044, coord[1] * 4 / 0.044]
        # Get distance and angle to nearest post
        if (player[1] >= 0):
            yards_to_goal = distance(to_yard(player), to_yard([1, 0.044]))
            if (player[1] <= 0.044):
                angle_to_goal = 0
            else:
                angle_to_goal = abs(angle(player, [1, 0.044]))
        else:
            yards_to_goal = distance(to_yard(player), to_yard([1, -0.044]))
            if (player[1] >= -0.044):
                angle_to_goal = 0
            else:
                angle_to_goal = abs(angle(player, [1, -0.044]))
        # Get distance to nearest opponent
        closest_opponent = closest_player(player, team='right_team')
        yards_to_closest_opponent = distance(
            player, obs['right_team'][closest_opponent]) * 4 / 0.044
        # Set boolean for whether the player has space to shoot
        space = 1 if yards_to_closest_opponent >= 1 else 0
        # Logistic regression equation
        y = 0.337 - 0.157 * yards_to_goal - 0.022 * angle_to_goal + 0.799 * space
        # Probability of scoring
        return np.exp(y) / (np.exp(y) + 1)

    # Get the position of the players to be controlled
    # We always control left team (observations and actions
    # are mirrored appropriately by the environment).
    controlled_player_position = obs['left_team'][obs['active']]

    # Offense
    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:
        if obs["game_mode"] == 1:
            # KickOff - Make a short pass to the closest teammate
            return direction_check(Action.ShortPass,
                                   coordinate_to_direction(controlled_player_position,
                                                           closest_player(controlled_player_position)))
        elif obs["game_mode"] == 2:
            # GoalKick
            return Action.LongPass
        elif obs["game_mode"] == 3:
            # FreeKick
            return Action.Shot
        elif obs["game_mode"] == 4:
            # Corner
            return Action.HighPass
        elif obs["game_mode"] == 5:
            # ThrowIn
            return Action.ShortPass
        elif obs["game_mode"] == 6:
            # Penalty
            return Action.Shot
        else:
            # Normal
            # Shoot when expected goal is quite high
            if expected_goal(controlled_player_position) > 0.05:
                return Action.Shot
            # Run towards the goal otherwise.
            return Action.Right
    # Defense
    else:
        # Run towards the ball.
        if obs['ball'][0] > controlled_player_position[0] + 0.05:
            return Action.Right
        if obs['ball'][0] < controlled_player_position[0] - 0.05:
            return Action.Left
        if obs['ball'][1] > controlled_player_position[1] + 0.05:
            return Action.Bottom
        if obs['ball'][1] < controlled_player_position[1] - 0.05:
            return Action.Top
        # Try to take over the ball if close to the ball.
        return Action.Slide
