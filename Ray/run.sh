CUDA_VISIBLE_DEVICES=0 python3 fsdp_main.py --rank 0 --world_size 2 &
CUDA_VISIBLE_DEVICES=1 python3 fsdp_main.py --rank 1 --world_size 2 &