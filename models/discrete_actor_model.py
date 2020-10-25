import torch
from rlpyt.models.mlp import MlpModel
import torch.nn.functional as F
from torch.distributions import Categorical


class DiscreteActorModel(torch.nn.Module):
    """DQN with vector input an MLP for Q-value outputs for
    the action set.
    """

    def __init__(self,
            input_shape,
            output_size,
            hidden_sizes=[256, 256],
            action_mask=True):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.head = MlpModel(input_shape, hidden_sizes, output_size)
        self.action_mask = action_mask

    def forward(self, observation, prev_action, prev_reward):
        """
        Takes in input observation and outputs all possible Q values of actions
        """
        obs = observation.type(torch.float)  # Expect torch.uint8 inputs
        out_size = self.head._output_size

        # For action masking
        if len(obs.shape) > 1:
            # Batch obs
            action_mask = obs[:, -out_size:].type(torch.bool)
        else:
            # Non batched obs
            action_mask = obs[-out_size:].type(torch.bool)

        # Action mask holds all values that are possible
        pi = self.head(obs)
        pi[~action_mask] = -10e9

        action_probs = F.softmax(pi, dim=-1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample()

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs
