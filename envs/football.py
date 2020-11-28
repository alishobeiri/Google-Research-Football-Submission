from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.samplers.collections import TrajInfo
from kaggle_environments import make
import gym
from observations.ObsParser import ObsParser
from observations.EgoCentric import EgoCentricObs
import numpy as np

from kaggle_environments.envs.football.helpers import Action

count = 0


class FootballEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 rank=0,
                 scenario_name="11_vs_11_kaggle",
                 right_agent="submission.py",
                 debug=False,
                 configuration=dict()):
        super(FootballEnv, self).__init__()
        global count
        # Randomly select handmade defense or builtin ai to add variance
        random = np.random.random()
        right_agent = "self_play.py" # "builtin_ai" if rank % 2 == 0 else "submission.py"
        print("Right agent: ", right_agent)
        self.agents = [None, right_agent]  # We will step on the None agent
        self.env = make("football",
                        debug=False,
                        configuration=configuration)
        self.obs_parser = EgoCentricObs()
        # Create action spaces
        self.action_space = gym.spaces.Discrete(len(Action))
        self.env_id = count
        count += 1
        obs = self.reset()
        # Maybe can build custom observation parsers

        self.observation_space = gym.spaces.Box(float('-inf'), float('inf'), obs.shape)

    def step(self, action):
        obs, reward, done, info = self.trainer.step([action])
        obs = obs['players_raw'][0]
        state, (l_score, r_score, custom_reward) = self.obs_parser.parse(obs, action)
        info['l_score'] = l_score
        info['r_score'] = r_score
        return state, custom_reward, done, info

    def reset(self):
        self.trainer = self.env.train(self.agents)
        obs = self.trainer.reset()
        self.obs_parser.reset()
        obs = obs['players_raw'][0]
        state, _ = self.obs_parser.parse(obs, None)
        return state

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        pass


def football_env(rank=0, **kwargs):
    return GymEnvWrapper(FootballEnv(rank, **kwargs), act_null_value=0)


class FootballTrajInfo(TrajInfo):
    """TrajInfo class for use with Football Env, to store raw game score separate
    from clipped reward signal."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.l_score = 0
        self.r_score = 0
        self.action = 0
        self.player_pos_x = 0
        self.player_pos_y = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.action = action
        self.l_score = env_info[0]
        self.r_score = env_info[1]
        self.player_pos_x = observation[0]
        self.player_pos_y = observation[1]
