from kaggle_environments.envs.football.helpers import *

state = {
    "prev_owner": None
}

@human_readable_agent
def agent(obs):
    player_pos = obs['left_team'][obs['active']]

    def move_to_position(pos):
        '''Uses the best action to move to a position in greedy fashion'''
        pass

    def get_player_on_ball(pos):
        '''Find which opposition player is on the ball'''
        pass
    prev_owner = state["prev_owner"]
    ball_owned_team = obs["ball_owned_team"]
    state["prev_owner"] = ball_owned_team
    if ball_owned_team == -1:
        if (prev_owner == 1 and
            prev_owner != ball_owned_team):
            # Possible options are, opposition just did:
                # Pass - Find ball path, check if there is friendly player in the path of the ball
                # Shot
                # Continued possession
            pass
        print("Indeterminate")
        return Action.Idle
    elif ball_owned_team == 0:
        print("Attack")
        pass
    elif ball_owned_team == 1:
        # Player who is active on the ball is likely heading
        # the same direction as the ball
        print("Defense")
        pass

    print("Current owner: ", ball_owned_team)

    return Action.Idle


