
"""
DQN in asynchronous mode with GPU sampler.  
(Or could use alternating GPU sampler).
"""
from rlpyt.envs.gym_schema import GymEnvWrapper
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.logging.context import logger_context

from envs.football import football_rlpyt_env


def build_and_train(scenario="11_vs_11_kaggle", run_id=0):
    # Change these inputs to match local machine and desired parallelism.
    affinity = make_affinity(
        run_slot=0,
        n_cpu_core=2,  # Use 16 cores across all experiments.
        n_gpu=2,  # Use 8 gpus across all experiments.
        gpu_per_run=1,
        sample_gpu_per_run=1,
        async_sample=True,
        optim_sample_share_gpu=False,
        # hyperthread_offset=24,  # If machine has 24 cores.
        # n_socket=2,  # Presume CPU socket affinity to lower/upper half GPUs.
        # gpu_per_run=2,  # How many GPUs to parallelize one run across.
        # cpu_per_run=1,
    )

    sampler = AsyncGpuSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,
        env_kwargs=dict(game="pong"),
        batch_T=4,
        batch_B=1,
        max_decorrelation_steps=1,
        eval_env_kwargs=dict(game="pong"),
        eval_n_envs=2,
        eval_max_steps=int(10e3),
        eval_max_trajectories=4,
    )
    algo = DQN(
        replay_ratio=8,
        min_steps_learn=1e4,
        replay_size=int(1e5)
    )
    agent = AtariDqnAgent()
    runner = AsyncRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=2e6,
        log_interval_steps=1e4,
        affinity=affinity,
    )
    config = dict(use_summary_writer=True)
    name = "football_dqn_" + scenario
    log_dir = "async_dqn"
    with logger_context(log_dir, run_id, name, config, use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', help='Football scenario', default='11_vs_11_kaggle')
    parser.add_argument('--run_id', help='run identifier (logging)', type=int, default=0)
    args = parser.parse_args()
    build_and_train(
        scenario=args.scenario,
        run_id=args.run_id,
    )
