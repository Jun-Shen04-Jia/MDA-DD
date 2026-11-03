export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python MDA-DD.py --dataset coco --num_queries 500 \
--rho 2 --lamda 0.1 --wandb
