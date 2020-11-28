from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.samplers.collections import TrajInfo
from kaggle_environments import make
import gym
from observations.ObsParser import ObsParser
from observations.EgoCentric import EgoCentricObs
import numpy as np

from kaggle_environments.envs.football.helpers import Action



class FootballSelfPlayEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 rank=0,
                 scenario_name="11_vs_11_kaggle",
                 right_agent="submission.py",
                 debug=False,
                 configuration=dict()):
        super(FootballSelfPlayEnv, self).__init__()
        # Randomly select handmade defense or builtin ai to add variance
        self.agents = [None, right_agent]  # We will step on the None agent
        self.env = make("football",
                        debug=False,
                        configuration=configuration)
        self.obs_parser = [EgoCentricObs() for i in range(2)]
        # Create action spaces
        self.action_space = gym.spaces.Tuple((
                                gym.spaces.Discrete(len(Action)),
                                gym.spaces.Discrete(len(Action))))
        obs = self.reset()
        # Maybe can build custom observation parsers

        self.observation_space = gym.spaces.Box(float('-inf'), float('inf'), obs[0].shape)

    def step(self, actions):
        actions = [[act] for act in actions]
        l_agent, r_agent = self.env.step(actions)
        done = l_agent['status'] == 'DONE'
        info = l_agent['info']
        obs = [l_agent, r_agent]
        state_list = []
        reward_list = []
        for i in range(len(self.obs_parser)):
            i_obs = obs[i]['observation']['players_raw'][0]
            state, (l_score, r_score, custom_reward) = self.obs_parser[i].parse(i_obs, None)
            state_list.append(state)
            reward_list.append(custom_reward)

        info['l_score'] = r_score # Since the right agent is processed last, the r_score is actually l_score
        info['r_score'] = l_score
        info['traj_done'] = done
        return np.array(state_list), np.array(reward_list), np.array([done, done]), info

    def reset(self):
        obs = self.env.reset()
        state_list = []
        reward_list = []
        for i in range(len(self.obs_parser)):
            self.obs_parser[i].reset()
            i_obs = obs[i]['observation']['players_raw'][0]
            state, (l_score, r_score, custom_reward) = self.obs_parser[i].parse(i_obs, None)
            state_list.append(state)
            reward_list.append((l_score, r_score, custom_reward))

        return state_list

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        pass


class FootballEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 rank=0,
                 scenario_name="11_vs_11_kaggle",
                 right_agent="submission.py",
                 debug=False,
                 configuration=dict()):
        super(FootballEnv, self).__init__()
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

def football_self_play_env(rank=0, **kwargs):
    return GymEnvWrapper(FootballSelfPlayEnv(rank, **kwargs), act_null_value=0)

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


class FootballSelfPlayTrajInfo(TrajInfo):
    """TrajInfo class for use with Football Env, to store raw game score separate
    from clipped reward signal."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_diff = 0
        self.left_reward = 0
        self.right_reward = 0
        self.left_score = 0
        self.right_score = 0
        self.left_action = []
        self.right_action = []
        self.left_pos_x = []
        self.left_pos_y = []
        self.right_pos_x = []
        self.right_pos_y = []

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.left_action += [action[0]]
        self.right_action += [action[1]]

        self.left_reward += reward[0]
        self.right_reward += reward[1]

        # Plot score after episode ends
        self.score_diff = env_info[0] - env_info[1]
        self.left_score = env_info[0]
        self.right_score = env_info[1]

        self.left_pos_x += [observation[0][0]]
        self.left_pos_y += [observation[0][1]]
        self.right_pos_x += [observation[1][0]]
        self.right_pos_y += [observation[1][1]]

        if env_info[2]:
            self.left_pos_x = np.array(self.left_pos_x).mean()
            self.left_pos_y = np.array(self.left_pos_y).mean()
            self.right_pos_x = np.array(self.right_pos_x).mean()
            self.right_pos_y = np.array(self.right_pos_y).mean()
            self.left_action = np.array(self.left_action).mean()
            self.right_action = np.array(self.right_action).mean()