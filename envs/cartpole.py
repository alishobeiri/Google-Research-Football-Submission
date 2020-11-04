from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.samplers.collections import TrajInfo
from kaggle_environments import make
import gym
from observations.ObsParser import ObsParser
from observations.EgoCentric import EgoCentricObs
import numpy as np

from kaggle_environments.envs.football.helpers import Action

from gym.envs.classic_control.cartpole import CartPoleEnv


def cartpole_env(env_id=1, **kwargs):
    return GymEnvWrapper(CartPoleEnv(**kwargs), act_null_value=0)

