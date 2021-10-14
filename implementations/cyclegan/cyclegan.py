import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.cuda.amp import autocast, GradScaler

import smdistributed.dataparallel.torch.distributed as dist
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
dist.init_process_group()


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--datasetA", action='append', help="List of paths to domain A")
parser.add_argument("--datasetB", action='append', help="List of paths to domain B")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--enable_amp", type=int, default=0, help="Enable Automatic Mixed Precision")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
opt.enable_amp=bool(opt.enable_amp)
opt.world_size = dist.get_world_size()
opt.rank = rank = dist.get_rank()
opt.local_rank = local_rank = dist.get_local_rank()
opt.batch_size //= opt.world_size // int(os.environ['SM_NUM_GPUS'])
opt.batch_size = max(opt.batch_size, 1)
is_first_local_rank = local_rank == 0
is_first_rank = rank == 0

if is_first_rank:
    print(opt)

torch.cuda.set_device(local_rank)

image_save_dir = "/checkpoint/eval"
model_save_dir = "/checkpoint/model"
# Create sample and checkpoint directories
os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks, opt.enable_amp)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks, opt.enable_amp)
D_A = Discriminator(input_shape, opt.enable_amp)
D_B = Discriminator(input_shape, opt.enable_amp)

if cuda:
    G_AB = DDP(G_AB.cuda())
    G_BA = DDP(G_BA.cuda())
    D_A = DDP(D_A.cuda())
    D_B = DDP(D_B.cuda())
    G_AB.cuda(local_rank)
    G_BA.cuda(local_rank)
    D_A.cuda(local_rank)
    D_B.cuda(local_rank)
    criterion_GAN.cuda(local_rank)
    criterion_cycle.cuda(local_rank)
    criterion_identity.cuda(local_rank)

if opt.epoch != 0:
    # Load pretrained models    
    G_AB.load_state_dict(torch.load(f"{model_save_dir}/G_AB_{opt.epoch:04d}.pth"))
    G_BA.load_state_dict(torch.load(f"{model_save_dir}/G_BA_{opt.epoch:04d}.pth"))
    D_A.load_state_dict(torch.load(f"{model_save_dir}/D_A_{opt.epoch:04d}.pth"))
    D_B.load_state_dict(torch.load(f"{model_save_dir}/D_B_{opt.epoch:04d}.pth"))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

#Gradient Scaler
scaler = GradScaler(enabled=opt.enable_amp)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Initialize DataSet
dataset = ImageDataset(opt.datasetA, opt.datasetB, transforms_=transforms_, unaligned=True)
train_dataset = dataset.train()
# Training data loader
train_sampler = torch.utils.data.distributed.DistributedSampler(
                                                        train_dataset,
                                                        num_replicas=opt.world_size,
                                                        rank=rank
                                                        )
dataloader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    sampler=train_sampler,
)

if is_first_rank:
# Test data loader
    val_dataloader = DataLoader(
        dataset.test(),
        batch_size=5,
        shuffle=True,
        num_workers=1,
    )


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, f"{image_save_dir}/{batches_done:08d}.png", normalize=False)


# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.module.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.module.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        with autocast(enabled=opt.enable_amp):

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        with autocast(enabled=opt.enable_amp):
            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

        scaler.scale(loss_D_A).backward()
        scaler.step(optimizer_D_A)

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        with autocast(enabled=opt.enable_amp):
            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

        scaler.scale(loss_D_B).backward()
        scaler.step(optimizer_D_B)
        
        scaler.update()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        if is_first_rank:
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0 and is_first_rank:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0 and is_first_rank:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), f"{model_save_dir}/G_AB_{epoch:04d}.pth")
        torch.save(G_BA.state_dict(), f"{model_save_dir}/G_BA_{epoch:04d}.pth")
        torch.save(D_A.state_dict(), f"{model_save_dir}/D_A_{epoch:04d}.pth")
        torch.save(D_B.state_dict(), f"{model_save_dir}/D_B_{epoch:04d}.pth")
