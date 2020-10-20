import torch
from rlpyt.models.mlp import MlpModel


class FootballDqnModel(torch.nn.Module):
    """DQN with vector input an MLP for Q-value outputs for
    the action set.
    """

    def __init__(self,
            input_shape,
            output_size,
            fc_sizes=[64, 64]):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.head = MlpModel(input_shape, fc_sizes, output_size)

    def forward(self, observation, prev_action, prev_reward):
        """
        Takes in input observation and outputs all possible Q values of actions
        """
        obs = observation.type(torch.float)  # Expect torch.uint8 inputs

        q = self.head(obs)

        return q
