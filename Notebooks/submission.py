from kaggle_environments.envs.football.helpers import *

@human_readable_agent
def agent(obs):
    # Get the position of the players to be controlled    
    # We always control left team (observations and actions
    # are mirrored appropriately by the environment).
    controlled_player_pos = obs['left_team'][obs['active']]
    
    # Offense
    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:
        if obs["game_mode"] == 1:
            # KickOff
            return Action.ShortPass
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
            if controlled_player_pos[0] > 0.5:
                return Action.Shot
            # Run towards the goal otherwise.
            return Action.Right
    # Defense
    else:
        # Run towards the ball.
        if obs['ball'][0] > controlled_player_pos[0] + 0.05:
            return Action.Right
        if obs['ball'][0] < controlled_player_pos[0] - 0.05:
            return Action.Left
        if obs['ball'][1] > controlled_player_pos[1] + 0.05:
            return Action.Bottom
        if obs['ball'][1] < controlled_player_pos[1] - 0.05:
            return Action.Top
        # Try to take over the ball if close to the ball.
        return Action.Slide
    
