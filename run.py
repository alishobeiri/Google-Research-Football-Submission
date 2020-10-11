# Set up the Environment.
from kaggle_environments import make

env = make("football",
           debug=True,
           configuration={"save_video": False,
                          "scenario_name": "academy_run_to_score_with_keeper",
                          "running_in_notebook": False,
                          'dump_full_episodes': False,
                          "render": True,
                          "logdir": "./logs"})

output = env.run(["submission.py", "builtin_ai"])[-1]
print(
    'Left player: reward = %s, status = %s, info = %s' % (output[0]['reward'], output[0]['status'], output[0]['info']))
print(
    'Right player: reward = %s, status = %s, info = %s' % (output[1]['reward'], output[1]['status'], output[1]['info']))
