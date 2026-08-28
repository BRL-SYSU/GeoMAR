export CXX=g++
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

conf_name='GeoMAR'

ROOT_PATH='./experiments/' # The path for saving model and logs

gpus='0,3,4,5,'
# gpus='0,'

#P: pretrain SL: soft learning
node_n=1

LOG_DIR='./logs'
mkdir -p "$LOG_DIR"
timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
log_file="${LOG_DIR}/${conf_name}_${timestamp}.log"

nohup python -u main_GeoMAR.py \
--root-path "$ROOT_PATH" \
--base "configs/${conf_name}.yaml" \
-t True \
--gpus "$gpus" \
--num-nodes "$node_n" \
> "$log_file" 2>&1 &
