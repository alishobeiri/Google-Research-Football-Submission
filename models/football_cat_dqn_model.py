import torch
from rlpyt.models.dqn.dueling import DuelingHeadModel, DistributionalDuelingHeadModel
from rlpyt.models.mlp import MlpModel
import torch.nn.functional as F


class FootballDqnModel(torch.nn.Module):
    """DQN with vector input an MLP for Q-value outputs for
    the action set.
    """

    def __init__(self,
            input_shape,
            output_size,
            fc_sizes=[128, 128, 128],
            dueling=False):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        if dueling:
            self.head = DuelingHeadModel(input_shape, fc_sizes, output_size)
        else:
            self.head = MlpModel(input_shape, fc_sizes, output_size)

    def forward(self, observation, prev_action, prev_reward):
        """
        Takes in input observation and outputs all possible Q values of actions
        """
        obs = observation.type(torch.float)  # Expect torch.uint8 inputs

        q = self.head(obs)

        return q

class DistributionalHeadModel(torch.nn.Module):
    """An MLP head which reshapes output to [B, output_size, n_atoms]."""

    def __init__(self, input_size, layer_sizes, output_size, n_atoms):
        super().__init__()
        self.mlp = MlpModel(input_size, layer_sizes, output_size * n_atoms)
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        return self.mlp(input).view(-1, self._output_size, self._n_atoms)


class FootballCatDqnModel(torch.nn.Module):
    """2D conlutional network feeding into MLP with ``n_atoms`` outputs
    per action, representing a discrete probability distribution of Q-values."""

    def __init__(
            self,
            input_shape,
            output_size,
            fc_sizes=[128, 128, 128],
            n_atoms=51,
            dueling=False):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.dueling = dueling
        if dueling:
            self.head = DistributionalDuelingHeadModel(input_shape, fc_sizes,
                output_size=output_size, n_atoms=n_atoms)
        else:
            self.head = DistributionalHeadModel(input_shape, fc_sizes,
                output_size=output_size, n_atoms=n_atoms)

    def forward(self, observation, prev_action, prev_reward):
        """Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity."""
        obs = observation.type(torch.float)  # Expect torch.uint8 inputs

        p = self.head(obs)
        p = F.softmax(p, dim=-1)

        return p
