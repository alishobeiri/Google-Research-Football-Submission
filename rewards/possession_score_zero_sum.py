
def possession_score_reward(possession, l_score_change, r_score_change, l_score, r_score, done):
    rew = 0
    if possession:
        rew += 0.002
    else:
        rew -= 0.002

    if l_score_change:
        rew += 2
    elif r_score_change:
        rew -= 2

    return rew