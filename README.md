# gfootball

## Important information

- Install GRF: [Ref](https://github.com/google-research/football#quick-start)
- Observations [Ref](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md)
- Actions [Ref](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/football/helpers.py)
- Important coordinates for the Field
  - Bottom left/right corner of the field is located at [-1, 0.42] and [1, 0.42], respectively.
  - Top left/right corner of the field is located at [-1, -0.42] and [1, -0.42], respectively.
  - Left/right goal is located at -1 and 1 X coordinate, respectively. They span between -0.044 and 0.044 in Y coordinates.
- Expected Goals [Paper](https://www.researchgate.net/publication/240641737_Estimating_the_probability_of_a_shot_resulting_in_a_goal_The_effects_of_distance_angle_and_space)
  - y  =  0.377  -   0.159 * (distance)  -  0.022 * (angle)  +  0.799 * (space)
  - "From  this  equation  it  is  possible  to  estimate  the probability  of  a  shot  scoring, based  on  its  location  (quantified  by  distance  in  yards  and  angle  in  degrees)  and  by whether  or  not  the  person  taking  the  shot  had  space  (quantified  by  0  if  there  was  an opponent within  one  metre, and 1 if not)."
  - [Paper](https://cartilagefreecaptain.sbnation.com/2014/9/11/6131661/premier-league-projections-2014#methoderology)
  - [Paper](https://www.americansocceranalysis.com/home/2014/05/08/calculating-expected-goals-2-0)

## Helper Function

- [x] Closest Player
- [x] Distance to Entity
- [x] Angle to Entity
- [ ] Check if opponent is in the way of the pass
- [x] Return direction to an entity
- [x] Expected Goal
- [ ] Check sticky array for activated actions
