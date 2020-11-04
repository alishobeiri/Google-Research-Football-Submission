
import torch
import torch.nn.functional as F
from rlpyt.models.mlp import MlpModel

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel


class FootballFfModel(torch.nn.Module):
    """
    Feedforward model for Atari agents: a convolutional network feeding an
    MLP with outputs for action probabilities and state-value estimate.
    """

    def __init__(
            self,
            input_shape,
            output_size,
            hidden_sizes=[256, 256, 256]
            ):
        """Instantiate neural net module according to inputs."""
        super().__init__()
        self.head = MlpModel(input_shape, hidden_sizes, output_size)
        self.pi = torch.nn.Linear(self.head.output_size, output_size)
        self.value = torch.nn.Linear(self.head.output_size, 1)

    def forward(self, observation, prev_action, prev_reward):
        """
        Compute action probabilities and value estimate from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Convolution layers process as [T*B,
        *image_shape], with T=1,B=1 when not given.  Expects uint8 images in
        [0,255] and converts them to float32 in [0,1] (to minimize image data
        storage and transfer).  Used in both sampler and in algorithm (both
        via the agent).
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, 1)

        fc_out = self.head(observation.view(T * B, *obs_shape))
        pi = F.softmax(self.pi(fc_out), dim=-1)
        v = self.value(fc_out).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v
