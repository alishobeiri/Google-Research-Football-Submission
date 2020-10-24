from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from agents.football_dqn_agent import FootballMixin
from models.football_cat_dqn_model import FootballCatDqnModel
from models.football_dqn_model import FootballDqnModel


class FootballCatDqnAgent(FootballMixin, CatDqnAgent):

    def __init__(self, ModelCls=FootballCatDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

