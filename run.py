# Set up the Environment.
from kaggle_environments import make

env = make("football",
           debug=True,
           configuration={"save_video": False,
                          "scenario_name": "academy_counterattack_hard",
                          "running_in_notebook": False,
                          'dump_full_episodes': False,
                          "render": True,
                          "logdir": "./logs"})

output = env.run(["builtin_ai", "submission.py"])[-1]
print(
    'Left player: reward = %s, status = %s, info = %s' % (output[0]['reward'], output[0]['status'], output[0]['info']))
print(
    'Right player: reward = %s, status = %s, info = %s' % (output[1]['reward'], output[1]['status'], output[1]['info']))
