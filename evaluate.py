# Set up the Environment.
from kaggle_environments import make

env = make("football",
           debug=True,
           configuration={"save_video": False,
                          "scenario_name": "11_vs_11_kaggle",
                          "running_in_notebook": False,
                          'dump_full_episodes': False,
                          "render": False,
                          "logdir": "./logs"})

N = 100
reward = 0
for i in range(N):
    output = env.run(["submission.py", "builtin_ai"])[-1]
    reward += output[0]['reward']

print("Mean reward over {} episodes: {}".format(N, reward / N))
