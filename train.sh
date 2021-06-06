# CUDA_VISIBLE_DEVICES=[cuda_device_ids] python -m torch.distributed.launch --nproc_per_node=[n_gpus] train.py [--train_options] 
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 train.py --name chunk64 --wandb
# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 train.py --real_batch_size 2 --batch_size 8