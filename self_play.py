import torch
import numpy as np
from kaggle_environments.envs.football.helpers import *
from numpy import arctan2
import hashlib
import os
import warnings
import torch.nn as nn
from torch.distributions import Normal


def infer_leading_dims(tensor, dim):
    """Looks for up to two leading dimensions in ``tensor``, before
    the data dimensions, of which there are assumed to be ``dim`` number.
    For use at beginning of model's ``forward()`` method, which should
    finish with ``restore_leading_dims()`` (see that function for help.)
    Returns:
    lead_dim: int --number of leading dims found.
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    shape: tensor shape after leading dims.
    """
    lead_dim = tensor.dim() - dim
    assert lead_dim in (0, 1, 2)
    if lead_dim == 2:
        T, B = tensor.shape[:2]
    else:
        T = 1
        B = 1 if lead_dim == 0 else tensor.shape[0]
    shape = tensor.shape[lead_dim:]
    return lead_dim, T, B, shape


def restore_leading_dims(tensors, lead_dim, T=1, B=1):
    """Reshapes ``tensors`` (one or `tuple`, `list`) to to have ``lead_dim``
    leading dimensions, which will become [], [B], or [T,B].  Assumes input
    tensors already have a leading Batch dimension, which might need to be
    removed. (Typically the last layer of model will compute with leading
    batch dimension.)  For use in model ``forward()`` method, so that output
    dimensions match input dimensions, and the same model can be used for any
    such case.  Use with outputs from ``infer_leading_dims()``."""
    is_seq = isinstance(tensors, (tuple, list))
    tensors = tensors if is_seq else (tensors,)
    if lead_dim == 2:  # (Put T dim.)
        tensors = tuple(t.view((T, B) + t.shape[1:]) for t in tensors)
    if lead_dim == 0:  # (Remove B=1 dim.)
        assert B == 1
        tensors = tuple(t.squeeze(0) for t in tensors)
    return tensors if is_seq else tensors[0]


def friendly_player_dist_to_ball(obs):
    ball_pos = obs["ball"]
    active_index = obs['active']
    player_pos = obs['left_team'][active_index]

    dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(player_pos))
    return dist


def closest_defender_to_ball(obs):
    ball_pos = obs["ball"]
    closest_def_distance = np.min(np.linalg.norm(np.array(ball_pos[:2]) - np.array(obs['right_team']), axis=1))
    return closest_def_distance

def dist_to_goal_x(obs):
    ball_pos = obs["ball"]
    dist = np.linalg.norm(np.array(ball_pos[0]) - np.array(goal_pos[0]))
    return dist


def dist_to_goal_y(obs):
    ball_pos = obs["ball"]
    dist = np.linalg.norm(np.array(ball_pos[1]) - np.array(goal_pos[1]))
    return dist

def possession_score_reward(obs, possession, l_score_change, r_score_change, action, l_score, r_score, done):
    rew = 0

    ball_dist_to_goal = dist_to_goal(obs)

    ball_dist_to_goal_x = dist_to_goal_x(obs)
    ball_dist_to_goal_y = dist_to_goal_y(obs)

    ball_dist_to_goal_x = ball_dist_to_goal_x + 1e-10 if ball_dist_to_goal_x == 0 else ball_dist_to_goal_x
    ball_dist_to_goal_y = ball_dist_to_goal_y + 1e-10 if ball_dist_to_goal_y == 0 else ball_dist_to_goal_y

    ball_owned_team = obs['ball_owned_team']
    d_to_ball = friendly_player_dist_to_ball(obs)
    if possession:
        if ball_owned_team == -1:
            # While ball is travelling don't give higher rewards
            rew += 0.01
        else:
            rew += 0.02 * (1 - ball_dist_to_goal_x / 2) * (1 - ball_dist_to_goal_y / 0.42)
    else:
        rew -= 0.4 * (ball_dist_to_goal_x/2) * (ball_dist_to_goal_y/0.42) * d_to_ball

    if l_score_change:
        rew += 20
    elif r_score_change:
        rew -= 20

    return rew

# MoE
warnings.filterwarnings("ignore", category=UserWarning)

class MlpModel(torch.nn.Module):
    """Multilayer Perceptron with last layer linear.

    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list or None for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif hidden_sizes is None:
            hidden_sizes = []
        hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
            zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend([layer, nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            sequence.append(torch.nn.Linear(last_size, output_size))
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
            else output_size)

    def forward(self, input):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        return self.model(input)

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = MlpModel(input_size, hidden_size, output_size)

    def forward(self, x):
        out = self.model(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ResBlock, self).__init__()
        self.model = MlpModel(input_size, hidden_size, output_size)

    def forward(self, x):
        y = self.model(x)
        y = y + x
        return y


class ResNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_blocks):
        super(ResNet, self).__init__()
        self.blocks = []

        # For every block except last, add residual connection
        for i in range(num_blocks):
            self.blocks.append(ResBlock(input_size, hidden_size, input_size))

        # Last wont support residual connection as sizes don't match
        self.blocks.append(torch.nn.Linear(input_size, output_size))

        self.model = torch.nn.Sequential(*self.blocks)

    def forward(self, x):
        y = self.model(x)
        return y


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = list((gates > 0).sum(0).cpu().numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, hidden_size, latent_dim, output_size, num_experts, num_blocks=3, noisy_gating=True,
                 k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.latent_size = latent_dim
        self.hidden_size = hidden_size
        self.k = k

        action_size = output_size

        input_size = input_size - action_size # Remove the action masking from the input to match sizes properly

        self.encoder = ResNet(input_size=input_size, hidden_size=hidden_size,
                              output_size=latent_dim, num_blocks=num_blocks)
        # instantiate experts
        self.experts = nn.ModuleList([ResNet(input_size=latent_dim,
                                             hidden_size=hidden_size,
                                             output_size=output_size,
                                             num_blocks=num_blocks)
                                      for i in range(self.num_experts)])
        self.value = ResNet(input_size=input_size, hidden_size=hidden_size,
                            output_size=1, num_blocks=num_blocks)
        self.w_gate = nn.Parameter(torch.zeros(latent_dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(latent_dim, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, observation, prev_action, prev_reward):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        train = self.training
        observation = observation.float()

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, 1)
        observation = observation.view(T * B, *obs_shape)
        action_mask = observation[:, -19:].type(torch.bool)
        observation = observation[:, :-19]

        z = self.encoder(observation)
        gates, load = self.noisy_top_k_gating(z, train)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(z)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        value = self.value(observation).squeeze(-1)
        y[~action_mask] = -1e24
        y = nn.functional.softmax(y, dim=-1)
        y, value = restore_leading_dims((y, value), lead_dim, T, B)
        return y, value

    def loss(self, observation, prev_action, prev_reward, loss_coef=1e-1):
        train = self.training
        observation = observation.float()

        lead_dim, T, B, obs_shape = infer_leading_dims(observation, 1)
        observation = observation.view(T * B, *obs_shape)
        action_mask = observation[:, -19:].type(torch.bool)
        observation = observation[:, :-19]

        z = self.encoder(observation)
        gates, load = self.noisy_top_k_gating(z, train)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        return loss


goal_pos = [1.0, 0]

sticky_action_lookup = {val.name: i for i, val in enumerate(sticky_index_to_action)}


def dist_to_goal(obs):
    ball_pos = obs["ball"]
    dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(goal_pos))
    return dist


def dist_between_points(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def angle_between_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Need to scale all x values down
    angle = arctan2(float(dy), float(dx))
    return angle


def valid_team(team):
    return team == 'WeKick'


# Shoot, short pass, long pass, high pass
inter_action_vec_lookup = {Action.Shot.value: 0, Action.ShortPass.value: 1,
                           Action.LongPass.value: 2, Action.HighPass.value: 3}


class EgoCentricObs(object):
    def __init__(self):
        self.constant_lookup = dict(
            prev_team=-1,
            intermediate_action_vec=[0, 0, 0, 0],
            possession=False,
            prev_l_score=0,
            prev_r_score=0
        )

    def reset(self):
        self.constant_lookup = dict(
            prev_team=-1,
            intermediate_action_vec=[0, 0, 0, 0],
            possession=False,
            prev_l_score=0,
            prev_r_score=0
        )

    def action_mask(self, obs):
        # We want to prevent certain actions from being taken by appending a binary vector that
        # indicates which actions are possible
        stick_actions = obs["sticky_actions"]
        action_mask = np.ones(len(Action))

        action_mask[Action.ReleaseDribble.value] = 0
        action_mask[Action.ReleaseSprint.value] = 0
        action_mask[Action.ReleaseDirection.value] = 0

        # Sliding is over used so prevent it from happening
        action_mask[Action.Slide.value] = 0
        if obs['ball_owned_team'] == -1:
            # Can move, sprint, idle
            action_mask[Action.LongPass.value] = 0
            action_mask[Action.HighPass.value] = 0
            action_mask[Action.ShortPass.value] = 0
            action_mask[Action.Shot.value] = 0
            action_mask[Action.Dribble.value] = 0
        elif obs['ball_owned_team'] == 0:
            # Can do everything but slide
            action_mask[Action.Slide.value] = 0
        elif obs['ball_owned_team'] == 1:
            action_mask[Action.LongPass.value] = 0
            action_mask[Action.HighPass.value] = 0
            action_mask[Action.ShortPass.value] = 0
            action_mask[Action.Shot.value] = 0
            action_mask[Action.Dribble.value] = 0

        # Handle sticky actions
        if any([i in stick_actions for i in range(8)]):
            # Any directional input
            action_mask[Action.ReleaseDirection.value] = 1

        if Action.Sprint.value in stick_actions:
            action_mask[Action.ReleaseSprint.value] = 1

        if Action.Dribble.value in stick_actions:
            action_mask[Action.ReleaseDribble.value] = 1

        return action_mask

    def parse(self, obs, prev_action=None):
        active_index = obs['active']
        player_pos = obs['left_team'][active_index]
        player_vel = obs['left_team_direction'][active_index]
        player_tired_factor = obs['left_team_tired_factor'][active_index]

        active_player = np.array([player_pos[0], player_pos[1] / 0.42,
                                  *player_vel, player_tired_factor])

        teammates = []
        for i in range(len(obs["left_team"])):
            # We purposely repeat ourselves to maintain consistency of roles
            i_player_pos = obs['left_team'][i]
            i_player_vel = obs['left_team_direction'][i]
            i_player_tired_factor = obs['left_team_tired_factor'][i]
            i_dist = dist_between_points(player_pos, i_player_pos)
            i_vel_mag = np.linalg.norm(i_dist)
            i_vel_ang = arctan2(i_player_vel[1], i_player_vel[0])
            angle = angle_between_points(player_pos, i_player_pos)
            teammates.append([i_player_pos[0], i_player_pos[1] / 0.42,
                              i_dist, np.cos(angle), np.sin(angle),
                              i_vel_mag, np.cos(i_vel_ang), np.sin(i_vel_ang),
                              i_player_tired_factor])

        enemy_team = []
        for i in range(len(obs["right_team"])):
            i_player_pos = obs['right_team'][i]
            i_player_vel = obs['right_team_direction'][i]
            i_player_tired_factor = obs['right_team_tired_factor'][i]
            i_dist = dist_between_points(player_pos, i_player_pos)
            i_vel_mag = np.linalg.norm(i_dist)
            i_vel_ang = arctan2(i_player_vel[1], i_player_vel[0])
            angle = angle_between_points(player_pos, i_player_pos)
            teammates.append([i_player_pos[0], i_player_pos[1] / 0.42,
                              i_dist, np.cos(angle), np.sin(angle),
                              i_vel_mag, np.cos(i_vel_ang), np.sin(i_vel_ang),
                              i_player_tired_factor])

        teammates = np.array(teammates).flatten()
        enemy_team = np.array(enemy_team).flatten()
        curr_dist_to_goal = dist_to_goal(obs)  # Closer distance have larger variance, farther less important

        # get other information
        game_mode = [0 for _ in range(7)]
        if (type(obs['game_mode']) is GameMode):
            game_mode[obs['game_mode'].value] = 1
        else:
            game_mode[obs['game_mode']] = 1

        sticky_action = [0 for _ in range(len(sticky_action_lookup))]
        if type(obs['sticky_actions']) is set:
            for action in obs['sticky_actions']:
                sticky_action[sticky_action_lookup[action.name]] = 1
        else:
            sticky_action = obs['sticky_actions']

        active_team = obs['ball_owned_team']
        prev_team = self.constant_lookup['prev_team']
        action_vec = self.constant_lookup['intermediate_action_vec']
        possession = False  # Determine if we have possession or not
        if ((active_team == 0 and prev_team == 0) or
                (active_team == 1 and prev_team == 0) or
                (active_team == 1 and prev_team == 1)):
            # Reset if lose the ball or keep the ball on pass
            self.constant_lookup['intermediate_action_vec'] = [0, 0, 0, 0]
            possession = False
        elif (active_team == -1 and prev_team == 0 and prev_action is not None):
            # Nobody owns right now and you had possession
            # Track prev actions
            if (type(prev_action) is Action and
                    prev_action.value in inter_action_vec_lookup):
                action_vec[inter_action_vec_lookup[prev_action.value]] = 1
            elif prev_action in inter_action_vec_lookup:
                action_vec[inter_action_vec_lookup[prev_action]] = 1
            possession = True

        if not possession and active_team == 0:
            possession = True

        if active_team != -1:
            self.constant_lookup['prev_team'] = active_team

        self.constant_lookup['possession'] = possession
        l_score, r_score = obs['score']
        prev_l_score, prev_r_score = self.constant_lookup['prev_l_score'], self.constant_lookup['prev_r_score']

        l_score_change = l_score - prev_l_score
        r_score_change = r_score - prev_r_score

        scalars = [obs['ball'][0],
                   obs['ball'][1] / 0.42,
                   *obs['ball_direction'],
                   obs['steps_left'] / 3000,
                   *game_mode,
                   curr_dist_to_goal,
                   *sticky_action,  # Tracks sticky actions
                   *action_vec,  # Tracks long term actions
                   l_score_change,
                   r_score_change,
                   possession,
                   active_team]

        scalars = np.r_[scalars].astype(np.float32)
        action_mask = self.action_mask(obs)
        combined = np.concatenate([active_player.flatten(), teammates.flatten(),
                                   enemy_team.flatten(), scalars.flatten(), action_mask.flatten()])
        done = False
        if obs['steps_left'] == 0:
            done = True

        reward = possession_score_reward(obs, possession, l_score_change, r_score_change, prev_action, l_score, r_score, done)

        self.constant_lookup['prev_r_score'] = r_score
        self.constant_lookup['prev_l_score'] = l_score

        return combined, (l_score, r_score, reward)


with open("pretrained/self_play/self_play.pkl", "rb") as f:
    old_checksum = hashlib.md5(f.read()).hexdigest()

state_dict = torch.load('pretrained/self_play/self_play.pkl')['agent_state_dict']
model = MoE(
    input_size=235 + 19,
    output_size=19,
    latent_dim=64,
    num_experts=10,
    hidden_size=[128, 128, 128],
    noisy_gating=True,
    k=4
)
model.load_state_dict(state_dict)
model.eval()
obs_parser = EgoCentricObs()
prev_action = None

@human_readable_agent
def agent(obs):
    global old_checksum
    global prev_action
    with open("pretrained/self_play/self_play.pkl", "rb") as f:
        checksum = hashlib.md5(f.read()).hexdigest()
    if old_checksum != checksum:
        old_checksum = checksum

        # Get latest 3 policies, randomly pick one
        files = sorted([os.path.join('pretrained/self_play', i) for i in os.listdir('pretrained/self_play/')])[-2:]
        index = np.random.randint(0, len(files))
        file = files[index]
        # Used to check if model has changed before loading new state dict
        state_dict = torch.load(file)
        if 'agent_state_dict' in state_dict:
            state_dict = state_dict['agent_state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        obs_parser.reset()

    model_inputs, _ = obs_parser.parse(obs, prev_action)
    obs_tensor = torch.from_numpy(model_inputs).float()
    pi, value = model(obs_tensor, None, None)
    sample = torch.multinomial(pi.view(-1, len(Action)), num_samples=1)
    sample = sample.cpu().detach().numpy().squeeze(-1).squeeze(-1)
    action = Action(sample)

    prev_action = action
    return action