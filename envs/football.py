from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.samplers.collections import TrajInfo
from kaggle_environments import make
import gym
from observations.ObsParser import ObsParser
from observations.EgoCentric import EgoCentricObs
import numpy as np

from kaggle_environments.envs.football.helpers import Action


class FootballEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 scenario_name="11_vs_11_kaggle",
                 right_agent="builtin_ai",
                 debug=False,
                 configuration=dict(),
                 env_id=0):
        super(FootballEnv, self).__init__()
        self.env_id = env_id
        self.agents = [None, right_agent]  # We will step on the None agent
        self.env = make("football",
                        debug=debug,
                        configuration=configuration)

        # Create action spaces
        self.action_space = gym.spaces.Discrete(len(Action))

        obs = self.reset()
        # Maybe can build custom observation parsers

        self.observation_space = gym.spaces.Box(float('-inf'), float('inf'), obs.shape)

    def step(self, action):
        obs, reward, done, info = self.trainer.step([action])
        obs = obs['players_raw'][0]
        state, (l_score, r_score, custom_reward) = EgoCentricObs.parse(obs)
        info['l_score'] = l_score
        info['r_score'] = r_score
        return state, custom_reward, done, info

    def reset(self):
        self.trainer = self.env.train(self.agents)
        obs = self.trainer.reset()
        obs = obs['players_raw'][0]
        state, _ = EgoCentricObs.parse(obs)
        return state

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        pass


def football_env(env_id=1, **kwargs):
    return GymEnvWrapper(FootballEnv(env_id=env_id, **kwargs), act_null_value=0)


class FootballTrajInfo(TrajInfo):
    """TrajInfo class for use with Football Env, to store raw game score separate
    from clipped reward signal."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = 0
        self.action = np.random.randn()
        self.observation = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.score = env_info[0]
        self.action = action
