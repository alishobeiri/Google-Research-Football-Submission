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
import os
import subprocess

import torch
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.algos.pg.ppo import PPO, PPOMoE
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector, GpuWaitResetCollector)
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.utils.logging.context import logger_context
from copy import deepcopy

from agents.football_ppo_agent import FootballFfAgent
from agents.football_ppo_moe_agent import FootballMoeAgent
from envs.cartpole import cartpole_env

from agents.football_cat_dqn_agent import FootballCatDqnAgent
from agents.football_dqn_agent import FootballDqnAgent
from agents.sac_discrete_agent import SACDiscreteAgent
from algos.cat_dqn import CategoricalDQNVector
from algos.dqn import DQNVector
from algos.sac_discrete import SACDiscrete
from envs.football import FootballTrajInfo, football_env

from rlpyt.utils.logging import logger
from rlpyt.utils.logging.context import LOG_DIR
from tensorboard import program

from models.football_cat_dqn_model import FootballCatDqnModel


def build_and_train(scenario="academy_empty_goal_close",
                    run_id=0,
                    log_interval_steps=int(10e5),
                    n_train_steps=10000,
                    cloud=False, bucket=None):
    env_kwargs = dict(debug=False,
                      configuration={
                          "save_video": False,
                          "scenario_name": scenario,
                          "running_in_notebook": False,
                          'dump_full_episodes': False,
                          "render": False,
                          "logdir": "./logs/test"}
                      )
    eval_kwargs = deepcopy(env_kwargs)
    eval_kwargs["configuration"]["render"] = False
    eval_kwargs["configuration"]["save_video"] = False
    # env_kwargs = dict()
    # eval_kwargs = env_kwargs.copy()
    run_async = False
    if run_async:
        affinity = make_affinity(
            run_slot=0,
            n_cpu_core=os.cpu_count(),  # Use 16 cores across all experiments.
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
    else:
        affinity = dict(workers_cpus=list(range(os.cpu_count())))
    state_dict = torch.load("pretrained/moe_resnet_latent_64_k_4_model_0.58618.pth")

    config = dict(
        # Batch T - How much samples to get before training, Batch B how many parallel to sample data
        # total data collected before each training iteration Batch_T * Batch_B
        algo=dict(
            # batch_size=batch_size,
            # replay_ratio=4,
            # min_steps_learn=int(0),
            # prioritized_replay=True,
            # double_dqn=True,
            # n_step_return=4,
            # V_min=-10,
            # V_max=10
        ),

        agent=dict(
            initial_model_state_dict=state_dict,
            # dueling=True
            # eps_itr_max=50000,
            model_kwargs=dict(
                latent_dim=64,
                num_experts=10,
                hidden_size=[128, 128, 128],
                noisy_gating=True,
                k=4
                # hidden_sizes=[128, 128, 128]
            )
        ),
        sampler=dict(batch_T=256, batch_B=os.cpu_count()),
    )
    sampler = CpuSampler(
        EnvCls=football_env,
        TrajInfoCls=FootballTrajInfo,
        env_kwargs=env_kwargs,
        eval_env_kwargs=eval_kwargs,
        max_decorrelation_steps=int(3000), # How many steps to take in env before training to randomize starting env state so experience isn't all the same
        eval_n_envs=100,
        eval_max_steps=int(100e5),
        eval_max_trajectories=200,
        **config["sampler"]  # More parallel environments for batched forward-pass.
    )

    algo = PPOMoE(**config["algo"])  # Run with defaults.
    agent = FootballMoeAgent(**config["agent"])
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=n_train_steps,
        log_interval_steps=log_interval_steps,
        affinity=affinity,
    )
    name = type(algo).__name__ + "_" + scenario + "_possession_scoring_reward"
    log_dir = 'training/' + name

    with logger_context(log_dir, run_id, name, config, snapshot_mode="gap", use_summary_writer=True):
        tb_loc = logger.get_tf_summary_writer().log_dir
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tb_loc, '--host', '0.0.0.0'])
        url = tb.launch()
        print("Tensorboard running at: ", url)
        # eval_kwargs["configuration"]["logdir"] = tb_loc
        runner.train()
        if cloud:
            storage_output_loc = tb_loc.strip("/")[len(LOG_DIR) + len('local'):].strip("/")
            subprocess.check_call([
                'gsutil', 'cp', '-r', tb_loc,
                os.path.join(bucket.strip("/"), storage_output_loc)])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', help='Football env scenario', default='11_vs_11_kaggle')
    parser.add_argument('--run_id', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--eval_max_trajectories', help='Max number of times to run a evaluation trajectory, \
                                                        helps to reduce variance', type=int, default=10)
    parser.add_argument('--log_interval_steps', help='Number of environment steps before logging', type=int, default=int(1e5))
    parser.add_argument('--batch_size', help='Batch size to train with', type=int, default=256)
    parser.add_argument('--n_train_steps', help='Number of environment steps to train on',
                        type=int, default=int(1e7))
    parser.add_argument('--min_steps_learn', help='Number of environment steps to take per parallel sampler before \
                                            training, default 256', type=int, default=0)# int(5e4))
    parser.add_argument('--batch_T', help='Number of environment steps to take per parallel sampler before \
                                            training', type=int, default=256)
    parser.add_argument('--cloud', help='Whether the project is on cloud or not', type=bool, default=False)
    parser.add_argument('--cloud_bucket', help='Storage bucket to save results to', type=str, default=None)

    args = parser.parse_args()

    if args.cloud and not args.cloud_bucket:
        raise IOError(
            "Please make sure that a storage bucket is defined for cloud training, use --cloud_bucket gs://kagglefootball-aiplatform for example")

    build_and_train(
        scenario=args.scenario,
        run_id=args.run_id,
        log_interval_steps=args.log_interval_steps,
        n_train_steps=args.n_train_steps,
        bucket=args.cloud_bucket,
        cloud=args.cloud
    )
