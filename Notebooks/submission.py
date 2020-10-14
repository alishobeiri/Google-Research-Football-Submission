from kaggle_environments.envs.football.helpers import *
import math
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


@human_readable_agent
def agent(obs):

    def distance(coord_one, coord_two):
        '''
        Compute the distance between two players
        Args:
            coord_one: Coordinates of the first entity (player, ball, goal ...)
            coord_two: Coordinates of the first entity
        Returns: Distance
        '''
        return np.linalg.norm(np.array(coord_one) - np.array(coord_two))

    def angle(coord_one, coord_two):
        '''
        Compute the angle between two coordinates
        Args:
            coord_one: Coordinates of the first entity (player, ball, goal ...)
            coord_two: Coordinates of the first entity
        Returns: Angle in degrees
        '''
        return np.degrees(np.arctan2(
            coord_two[1] - coord_one[1], coord_two[0] - coord_one[0]))

    def find_closest_opposition_to_ball():
        '''
        Find the opposing player closest to the ball
        Returns: Index of a opponent
        '''
        # Store distance to each opponent
        distances = obs['right_team'].copy()
        for i, player_position in enumerate(obs['right_team']):
            # Compute the distance between the ball and the player
            distances[i] = distance(obs['ball'], player_position)
        # Get the index for which the distance is minimum
        return np.argmin(distances)

    def closest_player(player_position, team='left_team'):
        '''
        Find the player closest to the player in posession
        Args:
            player_position: Coordinates of the player in posession
            team: name of the team in which we look for a player. Use team = 'right_team' to get the closest opponent and use team = 'left_team' to get the closest teammate
        Returns: Index of a player
        '''
        # Store distance to each player
        distances = obs[team].copy()
        for i, teammate_position in enumerate(obs[team]):
            # Compute the distance between the ball and the player
            dist = distance(player_position, teammate_position)
            # Set infinite distance for the distance between the player controlled and himself
            distances[i] = np.Inf if dist == 0 else dist
        # Get the index for which the distance is minimum
        return np.argmin(distances)

    def coordinate_to_direction(coord_one, coord_two):
        '''
        Find cardinal direction to get from an entity to another
        Args:
            coord_one: Coordinates of the first entity (player, ball, goal ...)
            coord_two: Coordinates of the first entity
        Returns: Cardinal direction
        '''
        # Compute angle between entities
        deg = angle(coord_one, coord_two)
        # Convert to a direction
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
        '''
        Check that the player controlled is directed properly before we let it perform an action
        Args:
            action: One of 19 actions for the player
            direction: One of 9 actions
        Returns: Cardinal direction
        '''
        # Sticky is a N hot encoded array of size 19 for whether the player is performing an action
        # Like passing, shooting, runing right or left ...
        if direction in obs['sticky_actions'][0:9]:
            # If the player is directed, perform an action
            return action
        else:
            return direction

    def expected_goal(player):
        '''
        Compute expected goals from paper :https://www.researchgate.net/publication/240641737_Estimating_the_probability_of_a_shot_resulting_in_a_goal_The_effects_of_distance_angle_and_space
        Args:
            player: Controlled player coordinates
        Returns: Float value representing probability of scoring
        '''
        # Football rules: size of goal : 8 yard by 8 feet
        def to_yard(coord): return [coord[0] * 4 / 0.044, coord[1] * 4 / 0.044]
        # Get distance and angle to nearest post
        if (player[1] >= 0):
            # Case where closer to the top post
            yards_to_goal = distance(to_yard(player), to_yard([1, 0.044]))
            # Set the angle to 0 when in front of goal
            if (player[1] <= 0.044):
                angle_to_goal = 0
            else:
                angle_to_goal = abs(angle(player, [1, 0.044]))
        else:
            # Case where closer to the bot post
            yards_to_goal = distance(to_yard(player), to_yard([1, -0.044]))
            # Set the angle to 0 when in front of goal
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

    def opponent_on_path(player, teammate):
        # Find if there is an opponent player on the path of the path
        if player[0] > teammate[0] and player[1] > teammate[1]:
            # Case 3: BottomLeft pass
            polygon = Polygon([(player[0] + body_radius * 2, player[1] - body_radius * 2),
                               (player[0] - body_radius * 2,
                                player[1] + body_radius * 2),
                               (teammate[0] - body_radius * 2,
                                teammate[1] + body_radius * 2),
                               (teammate[0] + body_radius * 2, teammate[1] - body_radius * 2)])

        elif player[0] < teammate[0] and player[1] > teammate[1]:
            # Case 2: BottomRight pass
            polygon = Polygon([(player[0] + body_radius * 2, player[1] + body_radius * 2),
                               (player[0] - body_radius * 2,
                                player[1] - body_radius * 2),
                               (teammate[0] - body_radius * 2,
                                teammate[1] - body_radius * 2),
                               (teammate[0] + body_radius * 2, teammate[1] + body_radius * 2)])

        elif player[0] > teammate[0] and player[1] < teammate[1]:
            # Case 4: TopLeft pass
            polygon = Polygon([(player[0] + body_radius * 2, player[1] + body_radius * 2),
                               (player[0] - body_radius * 2,
                                player[1] - body_radius * 2),
                               (teammate[0] - body_radius * 2,
                                teammate[1] - body_radius * 2),
                               (teammate[0] + body_radius * 2, teammate[1] + body_radius * 2)])

        elif player[0] < teammate[0] and player[1] < teammate[1]:
            # Case 1: TopRight pass
            polygon = Polygon([(player[0] - body_radius * 2, player[1] + body_radius * 2),
                               (player[0] + body_radius * 2,
                                player[1] - body_radius * 2),
                               (teammate[0] + body_radius * 2,
                                teammate[1] - body_radius * 2),
                               (teammate[0] - body_radius * 2, teammate[1] + body_radius * 2)])

        for opponent in enumerate(obs[team]):
            point = Point(opponent[0], opponent[1])
            if polygon.contains(point):
                return True
        return False

        def pass_path():
        pass

    def move_to_position(pos, player_pos):
        '''
        Calculates vector between pos and player_pos, calculates cosine similarity to
        find the best direction action to get the player to the desired location in
        straight line fashion
        Args:
            pos: Desired destination position
            player_pos: Current player position
        Returns: Direction action
        '''
        if (len(pos) == 3):
            # 3D vector, convert to 2D
            pos = pos[:2]
        dir_array = dir_lookup["vectors"]
        desired_dir = (np.array(pos) - np.array(player_pos)).reshape(1, -1)

        # Map direction to closest action using cosine similarity
        cosine_dist = 1 - cdist(desired_dir, dir_array, metric='cosine')
        max_index = np.argmax(cosine_dist)
        return dir_lookup["actions"][max_index]

    def on_breakaway(player_pos, obs):
        '''Detects if a player is on a breakaway or not'''
        player_x = player_pos[0]
        right_team_pos = obs["right_team"][1:]  # Not including the goalie
        return all([player_x > r_ply_pos[0] for r_ply_pos in right_team_pos])

    # Constants

    constants = {
        "delta_x": 9.2e-3,  # How much player / ball position changes when moving without sprinting
        # How much player / ball position changes when sprinting
        "sprinting_delta_x": 1.33e-2,
        "max_distance_to_influence_ball": 0.01,
        "max_distance_to_influence_ball_sprinting": 0.0185,
        "goalie_out_of_box_dist": 0.3,
        "on_breakaway_goalie_dist": 0.1
    }

    state = {
        "prev_owner": None,
        "count": 10,
        "prev_player_pos": [0, 0],
        "prev_ball_pos": [0, 0, 0]
    }

    dir_lookup = {
        "vectors": np.array([[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]]),
        "actions": [Action.Right, Action.TopRight, Action.Top, Action.TopLeft, Action.Left, Action.BottomLeft, Action.Bottom, Action.BottomRight]
    }

    # Games State
    player_pos = obs['left_team'][obs['active']]
    prev_owner = state["prev_owner"]
    ball_owned_team = obs["ball_owned_team"]
    state["prev_owner"] = ball_owned_team
    ball_pos = obs["ball"]
    ball_direction = obs["ball_direction"]
    ball_owned_player = obs["ball_owned_player"]
    goal_post_top = [1, 0.044]
    goal_post_bot = [1, -0.044]

    # Game Agent
    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint

    follow_ball_action = move_to_position(ball_pos, player_pos)

    # Offense
    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:
        if obs["game_mode"] == 1:
            # KickOff - Make a short pass to the closest teammate
            return direction_check(Action.ShortPass,
                                   coordinate_to_direction(player_pos,
                                                           closest_player(player_pos)))

        elif obs["game_mode"] == 2:
            # GoalKick - Make a short pass to the closest teammate or make a long shot
            teammate = closest_player(player_pos)
            if opponent_on_path(player_pos, teammate):
                return direction_check(Action.LongPass, Action.Right)
            else:
                return direction_check(Action.ShortPass,
                                       coordinate_to_direction(player_pos, teammate))

        elif obs["game_mode"] == 3:
            # FreeKick
            return direction_check(Action.Shot,
                                   coordinate_to_direction(player_pos, [1, 0]))

        elif obs["game_mode"] == 4:
            # Corner
            return direction_check(Action.HighPass,
                                   coordinate_to_direction(player_pos, [1, 0]))

        elif obs["game_mode"] == 5:
            # ThrowIn - Make a short pass to the closest teammate
            return direction_check(Action.ShortPass,
                                   coordinate_to_direction(player_pos,
                                                           closest_player(player_pos)))

        elif obs["game_mode"] == 6:
            # Penalty
            return direction_check(Action.Shot,
                                   coordinate_to_direction(player_pos, [1, 0]))

        else:
            # Normal
            # Calculate goalie position, if goalie presses could lead to loss of possesion
            goalie_pos = np.array(obs["right_team"][0])
            dist_to_goalie = np.linalg.norm(np.array(player_pos) - goalie_pos)
            dist_goalie_off_line = np.linalg.norm(
                np.array([1, 0]) - goalie_pos)

            # If we are on a breakaway and the goalie has come close, ideally we should move around goalie
            if on_breakaway(player_pos, obs):
                if dist_to_goalie < constants["on_breakaway_goalie_dist"]:
                    return Action.Shot

            # If goalie is really far out of the box worth trying a shot
            if dist_goalie_off_line > constants["goalie_out_of_box_dist"]:
                return Action.Shot

            # Shoot when expected goal is quite high
            if expected_goal(player_pos) > 0.05:
                return Action.Shot

            # Shot if we are 'close' to the goal (based on 'x' coordinate).
            if player_pos[0] > 0.7:
                return Action.Shot

            # Run towards the goal otherwise.
            return Action.Right
    # Defense
    else:
        # If the other team owns the ball
        opponent_pos = np.array(obs["right_team"][ball_owned_player])
        opponent_dir = np.array(obs["right_team_direction"][ball_owned_player])
        dist_to_ball = np.linalg.norm(opponent_pos - np.array(ball_pos[:2]))

        # If the player is close enough to change direction of the ball we should go to the new ball position
        if constants["max_distance_to_influence_ball_sprinting"] > dist_to_ball and np.linalg.norm(opponent_dir) != 0:
            # Need to make it a 2 vector as ball position 3 vector
            calc_ball_pos = np.array(ball_pos[:2])
            # Normalize their direction, opponent can change the ball dir by sprinting_delta_x each touch
            opponent_dir = opponent_dir / np.linalg.norm(opponent_dir)
            # Calculate how much the ball position will be changing in each direction
            ball_delta = constants["sprinting_delta_x"] * opponent_dir
            calc_ball_pos = calc_ball_pos + ball_delta

            # Return a new action that goes to where the ball should be going
            return move_to_position(calc_ball_pos, player_pos)
    state["prev_player_pos"] = player_pos
    state["prev_ball_pos"] = ball_pos

    # Just follow the ball
    return follow_ball_action
