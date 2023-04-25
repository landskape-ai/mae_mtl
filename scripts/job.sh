#!/bin/bash
#SBATCH --job-name=mae_mtl
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --signal=SIGUSR1@90 # 90 seconds before time limit
#SBATCH --output=/home/mila/d/diganta.misra/projects/mae_mtl/mae_mtl.out
#SBATCH --error=/home/mila/d/diganta.misra/projects/mae_mtl/mae_mtl.err

# exit_script() {
#     echo "Preemption signal, saving myself"
#     trap - SIGTERM # clear the trap
#     # Optional: sends SIGTERM to child/sub processes
#     kill -- -$$
# }

module load anaconda/3

conda activate /home/mila/d/diganta.misra/.conda/envs/mae

wandb login bd67cef57b7227730fe3edf96e11d954558a9d0d

ulimit -Sn $(ulimit -Hn)

lr=1e-4
fix_layer=12
config_file=/home/mila/d/diganta.misra/projects/mae_mtl/configs/vit/upernet_vit-b16_ln_640x640_160k_ade20k.py
pretrain=/home/mila/d/diganta.misra/scratch/mae_data/mae_pretrain_vit_base_mmcls.pth
data_root=/home/mila/d/diganta.misra/scratch/mae_data/ADEChallengeData2016
work_dir=/home/mila/d/diganta.misra/scratch/mae_data/ckpt
exp_dir=mae_fixB_layer${fix_layer}_lr${lr}_b16_640x640_ade20k


python -u tools/train.py ${config_file} \
    --work-dir ${work_dir}/${exp_dir} --seed 0 --deterministic \
    --cfg-options model.pretrained=$pretrain \
    model.backbone.fix_grad_backbone=${fix_layer} \
    data.samples_per_gpu=2 optimizer.lr=${lr} \
    data.train.data_root=$data_root \
    data.val.data_root=$data_root \
    data.test.data_root=$data_root
