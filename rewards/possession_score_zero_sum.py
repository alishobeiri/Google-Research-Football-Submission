
def possession_score_reward(possession, l_score_change, r_score_change, l_score, r_score, done):
    rew = 0
    if possession:
        rew += 0.002
    else:
        rew -= 0.002

    if done and l_score == r_score:
        # Draw
        rew += 1
    elif done and l_score > r_score:
        # Win
        rew += 5
    elif done and l_score < r_score:
        # Loss
        rew -= 5

    if l_score_change:
        rew += 1
    elif r_score_change:
        rew -= 1

    return rew