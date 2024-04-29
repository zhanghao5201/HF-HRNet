source ~/.poseshrc
conda activate mmpose


cd /mnt/petrelfs/zhanghao1/mmpose
pwd

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=$4
PORT=$5
RESUME=$6
PYTHONPATH="/mnt/petrelfs/zhanghao1/mmpose/src":$PYTHONPATH 
srun --partition=$PARTITION --quotatype=spot --async \
--mpi=pmi2 \
--gres=gpu:$GPUS \
--job-name=${JOB_NAME} \
--kill-on-bad-exit=1 \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $RESUME
