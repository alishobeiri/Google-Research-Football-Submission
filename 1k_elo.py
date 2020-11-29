
# --- Imports ---
from kaggle_environments.envs.football.helpers import *
import math
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString

# --- Constants ---
STEPS_RUN = 1/114
STEPS_SPRINT = 1/77
BODY_RADIUS = 0.012
GOAL_POST_BOT = [1, 0.03] #[1, 0.044]
GOAL_POST_TOP = [1, -0.03] #[1, -0.044]
GOAL_CENTER = [1, 0]

# --- Hyperparameters ---
SPRINT_RANGE = 0.6
LONG_SHOT_RANGE_X = 0.6
LONG_SHOT_RANGE_Y = 0.17
SHOOT_RANGE_X = 0.8
SHOOT_RANGE_Y = 0.15
CROSS_RANGE_X = 0.7
CROSS_RANGE_Y = 0.25
# XG_THRESHOLD = 0.1
PRESSURE_THRESHOLD = 2/77
GOALIE_OUT_OF_BOX = 0.2

# --- Helper Functions ---
## Geometry
def distance(coord_one, coord_two):
    ''' 
    Compute the distance between two coordinates 
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
    return np.degrees(np.arctan2(coord_two[1] - coord_one[1], coord_two[0] - coord_one[0]))

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
        return Action.BottomRight
    elif 67.5 < deg and deg <= 112.5:
        return Action.Bottom
    elif 112.5 <= deg and deg <= 157.5:
        return Action.BottomLeft
    elif deg > 157.5 or deg <= -157.5:
        return Action.Left
    elif deg <= -112.5:
        return Action.TopLeft
    elif deg <= -67.5:
        return Action.Top
    elif deg <= -22.5:
        return Action.TopRight

    return Action.Idle

def projection(ball, player_pos):
    '''
    Compute the projection of the player coordinates onto the line formed
    by the ball position and the goal coordinates
    Args:
        ball: Current ball position
        player_pos: Current player position
    Returns: Direction action
    '''
    point = Point([player_pos[0], player_pos[1]])
    line = LineString([(-1, 0),(ball[0], ball[1])])

    x = np.array(point.coords[0])
    u = np.array(line.coords[0])
    v = np.array(line.coords[len(line.coords)-1])
    n = v - u
    n /= np.linalg.norm(n, 2)
    return u + n*np.dot(x-u, n)

## Environment
def direction_check(action, direction, sticky):
    ''' 
    Check that the player controlled is directed properly before we let it perform an action
    Args:
        action: One of 19 actions for the player
        direction: One of 9 actions
        sticky: N hot encoded array of size 19 for whether the player is performing an action.
            Like passing, shooting, runing right or left ...
    Returns: Cardinal direction
    '''
    if direction in sticky:
        # If the player is directed, perform an action
        return action
    else:
        return direction

## Strategy
def closest_player(player_position, team, right=False):
    '''
    Find the player closest to the player in posession
    Args:
        player_position: Coordinates of the player in posession
        team: team in which we look for a player. 
            Use team = 'right_team' to get the closest opponent and use team = 'left_team' to get the closest teammate 
        right: Boolean for whether we look for players in front of the playerin possession
    Returns: Index of a player
    '''
    # Store distance to each player
    distances = team.copy()
    for i, other_position in enumerate(team):
        # Test if the teammate is to the right of the player
        if other_position[0] < player_position[0] and right:
            distances[i] = np.Inf
        else:
            # Compute the distance between the ball and the player
            dist = distance(player_position, other_position)
            # Set infinite distance for the distance between the player controlled and himself
            distances[i] = np.Inf if dist == 0 else dist
    # Get the index for which the distance is minimum
    return np.argmin(distances)

def closest_player_to_goal(player_position, team):
    ''' 
    Find the player closest to the goal
    Args:
        player_position: Coordinates of the player in posession
        team: team in which we look for a player. 
    Returns: Index of a player
    '''
    # Store distance to each player
    distances = team.copy()
    for i, teammate_position in enumerate(team):
        # Compute the distance between the goal and the player
        dist = distance(GOAL_CENTER, teammate_position)
        # Set infinite distance for the distance between the player controlled and himself
        distances[i] = np.Inf if distance(player_position, teammate_position) == 0 else dist
    # Get the index for which the distance is minimum
    return np.argmin(distances)

def last_player(player_position, team, striker=True):
    ''' 
    Find if the player controlled is the furthest striker or the last defender
    Args:
        team: team in which we look for player. 
        player_position: Coordinates of the player in posession
        striker: Boolean for whether we look for furthest striker or the last defender
    Returns: Boolean
    '''
    for teammate_position in team:
        # Last striker
        if striker and player_position[0] < teammate_position[0]:
            return False
        # Last defender
        if (not striker) and player_position[0] > teammate_position[0]:
            return False
    return True

def players_in_box(player_position, team):
    ''' 
    Args:
        player_position: Coordinates of the player in posession
        team: team in which we look for player. 
    Returns: Teammate indexes of those who are in the box
    '''
    players = []
    for i, teammate_position in enumerate(team):
        # Test if the teammate is in the box
        if teammate_position[0] > SHOOT_RANGE_X and abs(teammate_position[1]) < SHOOT_RANGE_Y:
            if distance(player_position, teammate_position):
                # test that we do not include hta player in possession 
                players.append(i)
    return players

def opponent_on_path(player, teammate, opponents):
    ''' 
    Find if there is an opponent player on the path of the path
    Args:
        player: Controlled player coordinates
        teammate: Teammate coordinates
        opponents: Opponent coordinates
    Returns: Boolean value for if there is a player obstructing
    '''
    if player[0] > teammate[0] and player[1] > teammate[1]:
        # Case 3: BottomLeft pass
        polygon = Polygon([(player[0] + BODY_RADIUS * 2, player[1] - BODY_RADIUS * 2), 
                           (player[0] - BODY_RADIUS * 2, player[1] + BODY_RADIUS * 2),
                           (teammate[0] - BODY_RADIUS * 2, teammate[1] + BODY_RADIUS * 2),
                           (teammate[0] + BODY_RADIUS * 2, teammate[1] - BODY_RADIUS * 2)])

    elif player[0] < teammate[0] and player[1] > teammate[1]:
        # Case 2: BottomRight pass
        polygon = Polygon([(player[0] + BODY_RADIUS * 2, player[1] + BODY_RADIUS * 2), 
                           (player[0] - BODY_RADIUS * 2, player[1] - BODY_RADIUS * 2),
                           (teammate[0] - BODY_RADIUS * 2, teammate[1] - BODY_RADIUS * 2),
                           (teammate[0] + BODY_RADIUS * 2, teammate[1] + BODY_RADIUS * 2)])

    elif player[0] > teammate[0] and player[1] < teammate[1]:
        # Case 4: TopLeft pass
        polygon = Polygon([(player[0] + BODY_RADIUS * 2, player[1] + BODY_RADIUS * 2), 
                           (player[0] - BODY_RADIUS * 2, player[1] - BODY_RADIUS * 2),
                           (teammate[0] - BODY_RADIUS * 2, teammate[1] - BODY_RADIUS * 2),
                           (teammate[0] + BODY_RADIUS * 2, teammate[1] + BODY_RADIUS * 2)])

    elif player[0] < teammate[0] and player[1] < teammate[1]:
        # Case 1: TopRight pass
        polygon = Polygon([(player[0] - BODY_RADIUS * 2, player[1] + BODY_RADIUS * 2), 
                           (player[0] + BODY_RADIUS * 2, player[1] - BODY_RADIUS * 2),
                           (teammate[0] + BODY_RADIUS * 2, teammate[1] - BODY_RADIUS * 2),
                           (teammate[0] - BODY_RADIUS * 2, teammate[1] + BODY_RADIUS * 2)])

    else:
        # Same coordinate
        return False

    for opponent in enumerate(opponents):
        point = Point(opponent[1][0], opponent[1][1])
        if polygon.contains(point):
            return True
    return False

def predict_ball_pos(ball_pos, ball_direction, sprint = True):
    '''
    Predict next step's ball position given its direction and speed
    Args:
        ball_pos: Ball coordinates
        ball_direction: Ball movement in all directions
        sprint: Boolean for whether the player in posession is sprinting. 
        Cannot get opponent sticky so assume allways running.
    Returns: predicted ball position
    '''
    if sprint:
        return [ball_pos[0] + (1+STEPS_SPRINT) * ball_direction[0], 
                ball_pos[1] + (1+STEPS_SPRINT) * ball_direction[1]]
    else:
        return [ball_pos[0] + (1+STEPS_RUN) * ball_direction[0], 
                ball_pos[1] + (1+STEPS_RUN) * ball_direction[1]]

def on_breakaway(player_pos, obs):
    '''Detects if a player is on a breakaway (aka 1v1 duel with goal) or not'''
    player_x = player_pos[0]
    right_team_pos = obs["right_team"][1:] # Not including the goalie
    return all([player_x > r_ply_pos[0] for r_ply_pos in right_team_pos])

def shoot_direction(ball_position):
    '''
    Find the direction to aim for the best spot on goal
    Args:
        ball_position: Coordinates of the ball
    Returns: Action.Direction
    '''
    if ball_position[1] > 0:
        return coordinate_to_direction(ball_position, GOAL_POST_BOT)
    else:
        return coordinate_to_direction(ball_position, GOAL_POST_TOP)
        
def highpass_direction(ball_position, team):
    team_x = [coord[0] for coord in team]
    return coordinate_to_direction(ball_position, team[np.argmax(team_x)])

def player_marked(player_pos, opponents):
    for opponent in opponents:
        if distance(player_pos, opponent) < BODY_RADIUS * 2:
            return True
    return False    

@human_readable_agent
def agent(obs):
    ## Games State
    player_pos = obs['left_team'][obs['active']]
    ball_owned_team = obs["ball_owned_team"]
    ball_pos = obs["ball"][:2]
    ball_direction = obs["ball_direction"]
    ball_owned_player = obs["ball_owned_player"]
    
    ## Game Agent
    if player_pos[0] < SPRINT_RANGE and Action.Sprint not in obs['sticky_actions']:
        # Sprint when in the final half of either side of the pitch 
        return Action.Sprint
    elif player_pos[0] > SPRINT_RANGE and Action.Sprint in obs['sticky_actions']:
        # Else run normally
        return Action.ReleaseSprint
    
    ## Game Modes
    if obs["game_mode"] == GameMode.KickOff:
        # KickOff - Make a short pass to the closest teammate
        return direction_check(Action.ShortPass, 
                               coordinate_to_direction(ball_pos, obs["left_team"][closest_player(player_pos, obs["left_team"])]),
                               obs['sticky_actions'])

    elif obs["game_mode"] == GameMode.GoalKick:
        # GoalKick - Make a short pass to the closest teammate or make a long shot 
        teammate = closest_player(player_pos, obs["left_team"])
        teammate_pos = obs["left_team"][teammate]
        if opponent_on_path(player_pos, teammate_pos, obs["right_team"]):
            # Choose a random place to highpass it
            return direction_check(Action.HighPass, highpass_direction(ball_pos, obs["left_team"]), obs['sticky_actions'])
        
        else:
            # ShortPass to a close teammate
            return direction_check(Action.ShortPass, 
                                   coordinate_to_direction(ball_pos,
                                                           teammate_pos), obs['sticky_actions'])

    elif obs["game_mode"] == GameMode.FreeKick:
        # FreeKick - Shoot when close to goal and pass ball otherwise
        # Shoot when close to goal
        if player_pos[0] > SHOOT_RANGE_X and abs(player_pos[1]) < SHOOT_RANGE_Y :
            return direction_check(Action.Shot,
                                   shoot_direction(ball_pos), obs['sticky_actions'])
        
        else:
            # When far from goal
            teammate = closest_player(player_pos, obs["left_team"])
            teammate_pos = obs["left_team"][teammate]            
            if opponent_on_path(player_pos, teammate_pos, obs["right_team"]) or player_marked(teammate_pos, obs['right_team']):
                # LongPass to the Right if the closest player is obstructed by opponent
                teammate = closest_player_to_goal(player_pos, obs["left_team"])
                teammate_pos = obs["left_team"][teammate]     
                return direction_check(Action.LongPass, 
                                       coordinate_to_direction(ball_pos,
                                                               teammate_pos), obs['sticky_actions'])
            else:
                # ShortPass to a close teammate
                return direction_check(Action.ShortPass,
                                       coordinate_to_direction(ball_pos,
                                                               teammate_pos), obs['sticky_actions'])
            
    elif obs["game_mode"] == GameMode.Corner:
        # Corner - Pass the ball to random points in the box
        directions = [coordinate_to_direction(ball_pos, GOAL_CENTER),
                      coordinate_to_direction(ball_pos, [.8, 0])]
        for direction in directions:
            if direction in obs['sticky_actions']:
                return Action.HighPass
        return np.random.choice(directions)
    
    elif obs["game_mode"] == GameMode.Penalty:
        # Penalty - Randomly choose a direction (either a goal post or the center of the goal)
        directions = [coordinate_to_direction(ball_pos, GOAL_CENTER),
                      coordinate_to_direction(ball_pos, GOAL_POST_TOP),
                      coordinate_to_direction(ball_pos, GOAL_POST_BOT)]
        for direction in directions:
            if direction in obs['sticky_actions']:
                return Action.Shot
        return np.random.choice(directions)
    
    # Offense - When we have the ball
    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:      
        # Only for ThrowIn Game Mode does the player have possession
        if obs["game_mode"] == GameMode.ThrowIn:
            # ThrowIn - Make a short pass to the closest teammate
            return direction_check(Action.ShortPass,
                                   coordinate_to_direction(player_pos,
                                                           obs["left_team"][closest_player(player_pos, obs["left_team"])]), obs['sticky_actions'])
        
        else:
            # Normal
            next_ball_pos = predict_ball_pos(ball_pos, ball_direction)
            if next_ball_pos[0] > 0.98:
                return Action.Left
            if next_ball_pos[1] > 0.40:
                return Action.Top
            if next_ball_pos[1] < -0.40:
                return Action.Bottom
            
            # When winning, waste time
            # if obs["score"][0] > obs["score"][1] and player_pos[0] < 0:
            #     teammate = closest_player(player_pos, obs["left_team"], right=True)
            #     teammate_pos = obs["left_team"][teammate]
            #     if opponent_on_path(player_pos, teammate_pos, obs["right_team"]) or player_marked(player_pos, obs["right_team"]):
            #         # LongPass to the Right if the closest player is obstructed by opponent
            #         teammate = closest_player_to_goal(player_pos, obs["left_team"])
            #         teammate_pos = obs["left_team"][teammate]
            #         return direction_check(Action.LongPass,
            #                                coordinate_to_direction(ball_pos,
            #                                                        teammate_pos), obs['sticky_actions'])
            #     else:
            #         # ShortPass to a close teammate
            #         return direction_check(Action.ShortPass,
            #                                coordinate_to_direction(ball_pos,
            #                                                        teammate_pos), obs['sticky_actions'])
                
            # Calculate goalie position
            goalie_pos = np.array(obs["right_team"][0])
            dist_to_goalie = np.linalg.norm(np.array(player_pos) - goalie_pos)
            dist_goalie_off_line = np.linalg.norm(np.array(GOAL_CENTER) - goalie_pos)
                        
            # Shoot when close to goal
            if player_pos[0] > SHOOT_RANGE_X and \
                abs(player_pos[1]) < SHOOT_RANGE_Y and \
                player_pos[0] < obs['ball'][0]:
                return direction_check(Action.Shot,
                                       shoot_direction(ball_pos), obs['sticky_actions'])

            # Clear the ball when close to our goal
            if player_pos[0] < -SHOOT_RANGE_X and \
                abs(player_pos[1]) < SHOOT_RANGE_Y:
                return direction_check(Action.Shot,
                                       coordinate_to_direction(player_pos, GOAL_CENTER), obs['sticky_actions'])
            
            # Shoot when goali leaves line and close-ish to goal
            if dist_goalie_off_line > GOALIE_OUT_OF_BOX and \
                player_pos[0] > LONG_SHOT_RANGE_X and \
                abs(player_pos[1]) < LONG_SHOT_RANGE_Y:
                return direction_check(Action.Shot,
                                       shoot_direction(ball_pos), obs['sticky_actions'])
  
            # if close to goal and too wide for shot pass the ball
            if player_pos[0] > CROSS_RANGE_X and abs(player_pos[1]) > CROSS_RANGE_Y:
                # Get players in box
                teammates = players_in_box(player_pos, obs["left_team"])
                for teammate in teammates:
                    teammate_pos = obs["left_team"][teammate]     
                    # If the player is not obstructed
                    if not opponent_on_path(player_pos, teammate_pos, obs["right_team"]):
                        # ShortPass to a that teammate
                        return direction_check(Action.ShortPass,
                                               coordinate_to_direction(ball_pos,
                                                                       teammate_pos), obs['sticky_actions'])
                    
            # Run towards the Right and along the sidelines
            if player_pos[0] < - 0.15:
                if player_pos[1] > 0:
                    return coordinate_to_direction(player_pos, [0, 0.3])
                else:
                    return coordinate_to_direction(player_pos, [0, -0.3])
            
            return coordinate_to_direction(player_pos, GOAL_CENTER)
        
    # Defense
    else:        
        # When the opponent has the ball
        opponent_pos = np.array(obs["right_team"][ball_owned_player])
        opponent_dir = np.array(obs["right_team_direction"][ball_owned_player])            
        dist_to_ball = np.linalg.norm(opponent_pos - np.array(ball_pos))
        
        # If the player is close enough to apply pressure and the ball is moving, then we should go to the new ball position
        if PRESSURE_THRESHOLD > dist_to_ball:
            return coordinate_to_direction(player_pos,
                                           predict_ball_pos(ball_pos,
                                                            ball_direction))
        
        # Run in the way of the player
        if player_pos[0] < opponent_pos[0] - BODY_RADIUS:
            return coordinate_to_direction(player_pos, projection(ball_pos, player_pos))

        # Run towards the ball
        return coordinate_to_direction(player_pos, ball_pos)

    # Default
    return coordinate_to_direction(player_pos, ball_pos)    
