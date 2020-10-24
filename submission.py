from kaggle_environments.envs.football.helpers import *
import numpy as np
from scipy.spatial.distance import cdist

constants = {
    "delta_x": 9.2e-3,  # How much player / ball position changes when moving without sprinting
    "sprinting_delta_x": 1.33e-2,  # How much player / ball position changes when sprinting
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


def find_closest_opposition_to_ball(ball_pos, active_players):
    '''Returns index of active player closest to the ball'''
    pass


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
    right_team_pos = obs["right_team"][1:] # Not including the goalie
    return all([player_x > r_ply_pos[0] for r_ply_pos in right_team_pos])


@human_readable_agent
def agent(obs):
    player_pos = obs['left_team'][obs['active']]
    prev_owner = state["prev_owner"]
    ball_owned_team = obs["ball_owned_team"]
    state["prev_owner"] = ball_owned_team
    ball_pos = obs["ball"]
    ball_direction = obs["ball_direction"]
    ball_owned_player = obs["ball_owned_player"]

    print(player_pos)

    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint

    follow_ball_action = move_to_position(ball_pos, player_pos)
    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:
        # Player is attacking

        # Calculate goalie position, if goalie presses could lead to loss of possesion
        goalie_pos = np.array(obs["right_team"][0])
        dist_to_goalie = np.linalg.norm(np.array(player_pos) - goalie_pos)
        dist_goalie_off_line = np.linalg.norm(np.array([1, 0]) - goalie_pos)

        # If we are on a breakaway and the goalie has come close, ideally we should move around goalie
        if on_breakaway(player_pos, obs):
            if dist_to_goalie < constants["on_breakaway_goalie_dist"]:
                return Action.Shot

        # If goalie is really far out of the box worth trying a shot
        if dist_goalie_off_line > constants["goalie_out_of_box_dist"]:
            return Action.Shot

        # Shot if we are 'close' to the goal (based on 'x' coordinate).
        if player_pos[0] > 0.5:
            return Action.Shot

        # Run towards the goal otherwise.
        return Action.Right
    elif ball_owned_team == 1:
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


