# %%
from collections import Counter
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import transformers 
from transformers import AdamW

import wandb
from tqdm.auto import tqdm
import plotly.graph_objects as go

from dataset import ClusteredModelNet40
from model.pcl import PclConfig, PclForClassification

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--real_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--wandb', action='store_true')
    return parser.parse_args()

args = get_args()

dist.init_process_group(backend='nccl', init_method='env://')
args.world_size = dist.get_world_size()
torch.cuda.set_device(args.local_rank)

# %%
if args.wandb and args.local_rank == 0:
    wandb.init(name=args.name, project='pcl', entity='hojun_cho_kaist')

# %%
dataset = ClusteredModelNet40(1024, 'train')
dataset_test = ClusteredModelNet40(1024, 'test')

# %%
config = PclConfig(num_labels=dataset.num_labels)
model = PclForClassification(config)
model.cuda()
model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

accumulation = args.batch_size // args.real_batch_size
sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank)
loader = DataLoader(dataset, batch_size=args.real_batch_size, pin_memory=True, sampler=sampler, num_workers=24)
optimizer = AdamW(model.parameters(), lr=args.lr)
num_training_steps = args.num_epochs * len(loader) // accumulation
lr_scheduler = transformers.get_scheduler(
    'cosine',
    optimizer=optimizer,
    num_warmup_steps=num_training_steps // 10,
    num_training_steps=num_training_steps,
)

sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test, num_replicas=args.world_size, rank=args.local_rank)
loader_test = DataLoader(dataset_test, batch_size=args.real_batch_size, pin_memory=True, sampler=sampler_test, num_workers=24)

# %%
if args.local_rank == 0: 
    progress_bar = tqdm(range(num_training_steps))

iteration = 0
for epoch in range(args.num_epochs):
    model.train()
    for input_points, cluster_ids, labels in loader:
        input_points, cluster_ids, labels = input_points.cuda(non_blocking=True), cluster_ids.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        outputs = model(input_points, cluster_ids, labels=labels)
        loss = outputs.loss / accumulation

        if iteration % accumulation == 0:
            optimizer.zero_grad(set_to_none=True)
            loss_avg = 0
            true_positive = 0
            label_size = 0

        loss.backward()
        loss_avg += loss.detach()
        true_positive += (outputs.logits.argmax(-1) == labels.squeeze(-1)).sum()
        label_size += labels.shape[0]

        if (iteration + 1) % accumulation == 0:
            acc_avg = true_positive / label_size
            handles = [dist.reduce(loss_avg, 0, async_op=True),
                       dist.reduce(acc_avg, 0, async_op=True)]

            optimizer.step()
            lr_scheduler.step()

            if args.wandb and args.local_rank == 0: 
                [handle.wait() for handle in handles]
                loss_avg /= args.world_size
                acc_avg /= args.world_size

                wandb.log({
                    'Train Loss': loss_avg.item(),
                    'Train Accuracy': acc_avg.item()
                }, step=(iteration + 1) // accumulation)
            
            if args.local_rank == 0:
                progress_bar.update(1)
            
            dist.barrier()

        iteration += 1
        del input_points, labels, outputs, loss

    # Evaluation
    model.eval()
    test_loss_avg = 0
    test_acc_avg = 0

    for input_points, cluster_ids, labels in loader_test:
        input_points, cluster_ids, labels = input_points.cuda(non_blocking=True), cluster_ids.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        with torch.no_grad():
            outputs = model(input_points, cluster_ids, labels=labels)
        test_loss_avg += outputs.loss
        test_acc_avg += (outputs.logits.argmax(-1) == labels.squeeze(-1)).sum() / labels.shape[0]

        del input_points, labels, outputs

    test_loss_avg /= len(loader_test)
    test_acc_avg = test_acc_avg / len(loader_test)
    handles = [dist.reduce(test_loss_avg, 0, async_op=True),
               dist.reduce(test_acc_avg, 0, async_op=True)]

    if args.wandb and args.local_rank == 0: 
        [handle.wait() for handle in handles]
        test_loss_avg /= args.world_size
        test_acc_avg /= args.world_size

        wandb.log({
            'Test Loss': test_loss_avg.item(),
            'Test Accuracy': test_acc_avg.item()
        }, step=(iteration + 1) // accumulation)
            
        model.module.save_pretrained('checkpoints/last_epoch')
    dist.barrier()
