#!/usr/bin/env bash
source ~/.poseshrc
conda activate mmpose

cd /mnt/petrelfs/zhanghao1/hf-hrnet/HF-HRNet
pwd

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=$4
PORT=$5
RESUME=$6
PYTHONPATH="/mnt/petrelfs/zhanghao1/hf-hrnet/HF-HRNet/src":$PYTHONPATH 
srun --partition=$PARTITION --quotatype=spot \
--mpi=pmi2 \
--gres=gpu:$GPUS \
--job-name=${JOB_NAME} \
--kill-on-bad-exit=1 \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --resume-from=$RESUME --launcher pytorch 


# bash tools/dist_train.sh Gveval-S1 pose_l_1 configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/hf_hrnet_18256_0315_19.py 8 20789
# bash tools/dist_train.sh Gveval-S1 pose_l_1 configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/hf_hrnet_30256_0315_19.py 8 20789

# bash tools/dist_train.sh Gveval-S1 pose_l_1 configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hf_hrnet_18256192_0315_19.py 8 20789
# bash tools/dist_train.sh Gveval-S1 pose_l_1 configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hf_hrnet_18384288_0315_19.py 8 20789
# bash tools/dist_train.sh Gveval-S1 pose_l_1 configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hf_hrnet_30256192_0315_19.py 8 20789
# bash tools/dist_train.sh Gveval-S1 pose_l_1 configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hf_hrnet_30384288_0315_19.py 8 20789
