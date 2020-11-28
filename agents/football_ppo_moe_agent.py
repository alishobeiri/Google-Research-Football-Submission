import torch
from rlpyt.agents.base import AgentStep, BaseAgent
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.agents.pg.base import AgentInfo
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to

from models.football_dqn_model import FootballDqnModel
from models.football_ff_model import FootballFfModel
from models.moe_model import MoE
import numpy as np

class FootballMixin:
    """
    Mixin class defining which environment interface properties
    are given to the model.
    """
    @staticmethod
    def make_env_to_model_kwargs(env_spaces):
        """Extract image shape and action size."""
        input_shape = env_spaces.observation.shape
        input_shape = env_spaces.observation.shape[0] if len(input_shape) == 1 else env_spaces.observation.shape
        return dict(input_size=input_shape,
                    output_size=env_spaces.action.n)

class FootballSelfPlayMixin:
    """
    Mixin class defining which environment interface properties
    are given to the model.
    """
    @staticmethod
    def make_env_to_model_kwargs(env_spaces):
        """Extract image shape and action size."""
        input_shape = env_spaces.observation.shape
        input_shape = env_spaces.observation.shape[0] if len(input_shape) == 1 else env_spaces.observation.shape
        return dict(input_size=input_shape,
                    output_size=19)


class FootballMoeAgent(FootballMixin, CategoricalPgAgent):

    def __init__(self, ModelCls=MoE, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class FootballMoeSelfPlayAgent(FootballSelfPlayMixin, BaseAgent):
    def __init__(self, ModelCls=MoE, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def __call__(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        pi, value = self.model(*model_inputs)
        return buffer_to((DistInfo(prob=pi), value), device="cpu")

    def initialize(self, env_spaces, share_memory=False,
                   global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
                           global_B=global_B, env_ranks=env_ranks)
        self.distribution = Categorical(dim=19)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        pi, value = self.model(*model_inputs)
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info, value=value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        _pi, value = self.model(*model_inputs)
        return value.to("cpu")
