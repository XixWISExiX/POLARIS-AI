#!/bin/bash
#
##SBATCH --partition=gpu,gpu_a100,disc_dual_a100_students,gpu_kepler,el9_gpu_test,sooner_gpu_test,sooner_gpu_test_ada,sooner_gpu_test_dual_ada,sooner_gpu_tes    t_quad_ada
##SBATCH --partition=gpu,gpu_a100,sooner_gpu_test
##SBATCH --partition=gpu_a100
##SBATCH --partition=debug_gpu
##SBATCH --partition=gpu
#SBATCH --partition=disc_dual_a100
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:a100:1 # NVIDIA A100 (any memory size).
##SBATCH --constraint=80gb # filter to 80 GB cards (features=80gb in sinfo).
##SBATCH --cpus-per-task=64
#SBATCH --cpus-per-task=32
##SBATCH --cpus-per-task=20
#SBATCH --mem=32G
##SBATCH --mem=45G
#SBATCH --output=results/ragen_%j_stdout.txt
#SBATCH --error=results/ragen_%j_stderr.txt
##SBATCH --time=12:00:00
#SBATCH --time=00:30:00
#SBATCH --job-name=MAIN-RAGEN-RUN
##SBATCH --mail-user=ADD YOUR OWN EMAIL
#SBATCH --mail-type=ALL
##SBATCH --chdir=/scratch/wiseman/POLARIS-AI/
#SBATCH --chdir=/scratch/wiseman/POLARIS-AI/run-experiments/slurm
##SBATCH --array=0-4    # the double ## means that this line is ignored

#################################################

## Environment Needs to be set up in some form!!!!

USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.ro        llout_filter_ratio=1"

# Line 45 in train_all.sh
python ../../external/RAGEN/train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=frozenlake-base-ppo algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1 $USE_PPO

