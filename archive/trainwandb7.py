# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
# go go go for MinM mask

import argparse
import datetime
import json
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from pathlib import Path
import fcntl

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms.functional as TF

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models import models_mae2 as models_mae

from engine.engine_pretrain2 import train_one_epoch
import wandb
wandb.login(key="9c1af0a383f6b1138e4ab20f65a4d6e4f194ffad")

progress_file = "./progress.txt"
def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/Sam2/sam2/imagenette', type=str,
                        help='dataset path')
    parser.add_argument('--patch_mask_path', default='/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/Sam2/sam2/imagenette/mask', type=str,
                        help='存储 .npy patch mask 的路径')
    parser.add_argument('--output_dir', default='output_dir5',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output_dir5',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, patch_mask_root, transform=None):
        """
        data_root: 原始图像路径
        patch_mask_root: 存放 patch 级 mask (.npy) 的路径
        transform: 图像变换
        """
        self.data_root = data_root
        self.patch_mask_root = patch_mask_root
        self.transform = transform

        # 只加载 JPEG 图像
        self.image_filenames = sorted([f for f in os.listdir(self.data_root) if f.endswith('.JPEG')])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # 1️⃣ 获取原图 & mask 文件名
        img_name = self.image_filenames[idx]  # e.g., "image001.JPEG"
        img_stem = os.path.splitext(img_name)[0]  # 去掉扩展名 -> "image001"

        img_path = os.path.join(self.data_root, img_name)  # 原图路径
        patch_mask_path = os.path.join(self.patch_mask_root, f"{img_stem}_patch_mask.npy")  # mask 路径

        # 2️⃣ 检查 mask 是否存在
        if not os.path.exists(patch_mask_path):
            raise FileNotFoundError(f"Patch mask file not found: {patch_mask_path}")

        # 3️⃣ 加载数据
        image = Image.open(img_path).convert("RGB")  # 加载原图
        patch_mask = np.load(patch_mask_path)  # 🎯 加载 patch mask (N, L)

        # 4️⃣ 变换数据
        if self.transform:
            image = self.transform(image)  # 对图像做变换
            patch_mask = torch.tensor(patch_mask, dtype=torch.float32)  # 🎯 转换为 Tensor

        return image, patch_mask  # 🎯 现在返回的 mask 是 `.npy`


def main(args):
    if misc.is_main_process():  
        wandb.init(
            project="MInM",
            entity="visual-intelligence-laboratory",  
            config=vars(args),
            name="MAE_imagenette_bs256_epoch400_noevaluate_sam"
        )

    args.distributed = False

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = MaskedDataset(
        data_root=os.path.join(args.data_path, 'train'),
        patch_mask_root=os.path.join(args.data_path, 'patch_masks'),  # 🎯 mask 现在存放在 `patch_masks/`
        transform=transform_train
    )

    print(dataset_train)

    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        sampler=sampler_train,
        drop_last=True,
        shuffle=True  # 🎯 这里确保训练数据正确打乱
    )

    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
    
        current_lr = optimizer.param_groups[0]['lr']
        if misc.is_main_process():
            wandb.log({
                'epoch': epoch,
                'train_loss': train_stats.get('loss', None),
                'learning_rate': current_lr,
            })
         
        if epoch % 10 == 4 or epoch + 1 == args.epochs:
            misc.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, name="temporary.pth"
                        )


            # if misc.is_main_process():
            #     if acc > best_accuracy:
            #         best_accuracy = acc
            #         if args.output_dir:
            #
            #             misc.save_model(
            #                 args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            #                 loss_scaler=loss_scaler, epoch=epoch, name="best_acc.pth"
            #             )
            #         wandb.save(os.path.join(args.output_dir, f"best_acc.pth"))

        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    wandb.log({'total_training_time': total_time}) 
    wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

