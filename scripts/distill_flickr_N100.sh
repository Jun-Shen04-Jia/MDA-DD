export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python MDA-DD.py --dataset flickr --num_queries 100 \
--rho 1 --lamda 0.5 --wandb
