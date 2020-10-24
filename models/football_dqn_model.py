import torch
from rlpyt.models.dqn.dueling import DuelingHeadModel
from rlpyt.models.mlp import MlpModel


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
