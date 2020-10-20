from rlpyt.agents.dqn.dqn_agent import DqnAgent
from models.football_dqn_model import FootballDqnModel


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
        return dict(input_shape=input_shape,
                    output_size=env_spaces.action.n)


class FootballDqnAgent(FootballMixin, DqnAgent):

    def __init__(self, ModelCls=FootballDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

