# Set up the Environment.
from kaggle_environments import make
from gfootball.env import create_environment

env_football = create_environment("11_vs_11_kaggle")

env = make("football",
           debug=True,
           configuration={"save_video": False,
                          "scenario_name": "academy_close_w_goalie_random",
                          "running_in_notebook": False,
                          'dump_full_episodes': False,
                          "render": True,
                          "logdir": "./logs"})

output = env.run(["submission.py", "builtin_ai"])[-1]
print(
    'Left player: reward = %s, status = %s, info = %s' % (output[0]['reward'], output[0]['status'], output[0]['info']))
print(
    'Right player: reward = %s, status = %s, info = %s' % (output[1]['reward'], output[1]['status'], output[1]['info']))
