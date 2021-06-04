# %%
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from transformers import AdamW
from transformers import get_scheduler

import wandb
from tqdm.auto import tqdm
import plotly.graph_objects as go

from dataset import ModelNet40
from model.pcl import PclConfig, PclForClassification

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--name', type=str, default='debug')
    return parser.parse_args()

args = get_args()

dist.init_process_group(backend='nccl', init_method='env://')
args.world_size = dist.get_world_size()
torch.cuda.set_device(args.local_rank)

# %%
if args.local_rank == 0:
    wandb.init(name=args.name, project='pcl', entity='hojun_cho_kaist')

# %%
dataset = ModelNet40(1024, 'train')
dataset_test = ModelNet40(1024, 'test')

# %%
config = PclConfig(num_labels=dataset.num_labels)
model = PclForClassification(config)
model.cuda()
model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank)
loader = DataLoader(dataset, batch_size=args.batch_size//args.accumulation, pin_memory=True, sampler=sampler)
optimizer = AdamW(model.parameters(), lr=args.lr)
num_training_steps = args.num_epochs * len(loader) // args.accumulation
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test, num_replicas=args.world_size, rank=args.local_rank)
loader_test = DataLoader(dataset_test, batch_size=args.batch_size//args.accumulation, pin_memory=True, sampler=sampler_test)

# %%
if args.local_rank == 0: 
    progress_bar = tqdm(range(num_training_steps))

iteration = 0
for epoch in range(args.num_epochs):
    model.train()
    for input_points, labels in loader:
        input_points, labels = input_points.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        outputs = model(input_points, labels=labels)
        loss = outputs.loss / args.accumulation

        if iteration % args.accumulation == 0:
            optimizer.zero_grad(set_to_none=True)
            loss_avg = 0
            true_positive = 0
            label_size = 0

        loss.backward()
        loss_avg += loss.detach()
        true_positive += (outputs.logits.argmax(-1) == labels).sum()
        label_size += labels.shape[0]

        if (iteration + 1) % args.accumulation == 0:
            acc_avg = true_positive / label_size
            handles = [dist.reduce(loss_avg, 0, async_op=True),
                       dist.reduce(acc_avg, 0, async_op=True)]

            optimizer.step()
            lr_scheduler.step()

            if args.local_rank == 0: 
                [handle.wait() for handle in handles]
                loss_avg /= args.world_size
                acc_avg /= args.world_size

                wandb.log({
                    'Train Loss': loss_avg.item(),
                    'Train Accuracy': acc_avg.item()
                }, step=(iteration + 1) // args.accumulation)
                progress_bar.update(1)
            
            dist.barrier()

        iteration += 1
        del input_points, labels, outputs, loss

    # Evaluation
    model.eval()
    test_loss_avg = 0
    test_true_positive = 0

    for input_points, labels in loader_test:
        input_points, labels = input_points.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        with torch.no_grad():
            outputs = model(input_points, labels=labels)
        test_loss_avg += outputs.loss
        test_true_positive += (outputs.logits.argmax(-1) == labels).sum()

        del input_points, labels, outputs

    test_loss_avg /= len(loader_test)
    test_acc_avg = test_true_positive / len(dataset_test)
    handles = [dist.reduce(test_loss_avg, 0, async_op=True),
               dist.reduce(test_acc_avg, 0, async_op=True)]

    if args.local_rank == 0: 
        [handle.wait() for handle in handles]
        test_loss_avg /= args.world_size
        test_acc_avg /= args.world_size

        wandb.log({
            'Test Loss': test_loss_avg.item(),
            'Test Accuracy': test_acc_avg.item()
        }, step=(iteration + 1) // args.accumulation)
            
        model.module.save_pretrained('checkpoints/last_epoch')
    dist.barrier()
