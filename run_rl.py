"""
Runs multiple instances of the Atari environment and optimizes using A2C
algorithm and a recurrent agent. Uses GPU parallel sampler, with option for
whether to reset environments in middle of sampling batch.

Standard recurrent agents cannot train with a reset in the middle of a
sequence, so all data after the environment 'done' signal will be ignored (see
variable 'valid' in algo).  So it may be preferable to pause those environments
and wait to reset them for the beginning of the next iteration.

If the environment takes a long time to reset relative to step, this may also
give a slight speed boost, as resets will happen in the workers while the master
is optimizing.  Feedforward agents are compatible with this arrangement by same
use of 'valid' mask.

"""
import torch
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector, GpuWaitResetCollector)
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.utils.logging.context import logger_context

from agents.football_dqn_agent import FootballDqnAgent
from algos.dqn import DQNVector
from envs.football import footbal_env, FootballTrajInfo


def build_and_train(scenario="academy_empty_goal_close", run_ID=0):

    env_kwargs = dict(debug=False,
                      configuration={
                          "save_video": False,
                          "scenario_name": scenario,
                          "running_in_notebook": False,
                          'dump_full_episodes': False,
                          "render": False,
                          "logdir": "./logs"}
                      )

    affinity = make_affinity(
        run_slot=0,
        n_cpu_core=8,  # Use 16 cores across all experiments.
        n_gpu=1,  # Use 8 gpus across all experiments.
        gpu_per_run=1,
        sample_gpu_per_run=1,
        async_sample=True,
        optim_sample_share_gpu=True,
        # hyperthread_offset=24,  # If machine has 24 cores.
        # n_socket=2,  # Presume CPU socket affinity to lower/upper half GPUs.
        # gpu_per_run=2,  # How many GPUs to parallelize one run across.
        # cpu_per_run=1,
    )
    config = dict(
        algo=dict(batch_size=256, replay_ratio=4, min_steps_learn=10),
        sampler=dict(batch_T=1024, batch_B=7),
    )
    sampler = AsyncGpuSampler(
        EnvCls=footbal_env,
        TrajInfoCls=FootballTrajInfo,
        env_kwargs=env_kwargs,
        eval_env_kwargs=env_kwargs,
        max_decorrelation_steps=int(0),
        eval_n_envs=5,
        eval_max_steps=int(10e3),
        eval_max_trajectories=10,
        **config["sampler"]  # More parallel environments for batched forward-pass.
    )

    algo = DQNVector(**config["algo"])  # Run with defaults.
    agent = FootballDqnAgent()
    runner = AsyncRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=int(1e24),
        log_interval_steps=1e5,
        affinity=affinity,
    )
    config = dict(scenario=scenario)
    name = "dqn_" + scenario
    log_dir = "example_4"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="gap", use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', help='Atari game', default='academy_empty_goal_close')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--mid_batch_reset', help='whether environment resets during itr',
                        type=bool, default=False)
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=1)
    args = parser.parse_args()
    build_and_train(
        scenario=args.scenario,
        run_ID=args.run_ID
    )
