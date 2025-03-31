# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torchvision.datasets as datasets

import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae2 as models_mae
from engine_probing2 import linear_probing
from engine_pretrain2 import train_one_epoch
import wandb

wandb.login(key="9c1af0a383f6b1138e4ab20f65a4d6e4f194ffad")


class MaskedImageDataset(Dataset):
    def __init__(self, image_root, mask_root, transform=None):
        """
        image_root: Root directory of training images, containing subdirectories (classes).
        mask_root: Root directory of masks, with corresponding subdirectories (classes).
        transform: Image transformations.
        """
        self.image_paths = []
        self.mask_paths = []
        self.class_indices = []
        self.transform = transform if transform else transforms.ToTensor()

        class_folders = sorted(os.listdir(image_root))
        print(f"Detected {len(class_folders)} classes: {class_folders}")

        for class_idx, class_name in enumerate(class_folders):
            image_class_dir = os.path.join(image_root, class_name)
            mask_class_dir = os.path.join(mask_root, class_name)

            if not os.path.isdir(image_class_dir) or not os.path.isdir(mask_class_dir):
                continue

            images = sorted([f for f in os.listdir(image_class_dir) if f.endswith(".JPEG")])
            masks = sorted([f for f in os.listdir(mask_class_dir) if f.endswith("_mask_applied.png")])

            for img_name in images:
                image_path = os.path.join(image_class_dir, img_name)
                mask_name = img_name.replace(".JPEG", "_mask_applied.png")
                mask_path = os.path.join(mask_class_dir, mask_name)

                if os.path.exists(mask_path):
                    self.image_paths.append(image_path)
                    self.mask_paths.append(mask_path)
                    self.class_indices.append(class_idx)
                else:
                    print(f"⚠️ Warning: Mask not found for {mask_path}")

        print(f"✅ Loaded {len(self.image_paths)} images and {len(self.mask_paths)} masks")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        fixed_image = Image.open("/data/home/zbz5349/WorkSpace/aigeeks/minm_mae/ImageNet-1K/train/n01530575/n01530575_5.JPEG").convert("RGB")
        fixed_mask_path = "/data/home/zbz5349/WorkSpace/aigeeks/minm_mae/ImageNet-1K/sam/masks_applied/n01530575/n01530575_5_mask_applied.png"
        fixed_mask = Image.open(fixed_mask_path).convert("L")
        class_idx = 0  # 固定类别

        fixed_mask = TF.to_tensor(fixed_mask)
        if fixed_mask.shape[0] == 1:
            fixed_mask = fixed_mask.repeat(3, 1, 1)

        if self.transform:
            fixed_image = self.transform(fixed_image)
            for t in self.transform.transforms:
                if not isinstance(t, transforms.ToTensor):
                    fixed_mask = t(fixed_mask)

        return fixed_image, fixed_mask_path, class_idx



def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-6)  # Adjusted for cosine annealing
    parser.add_argument('--warmup_epochs', type=int, default=40)
    parser.add_argument('--data_path', default='/data/home/zbz5349/WorkSpace/aigeeks/minm_mae/ImageNet-1K', type=str)
    parser.add_argument('--output_dir', default='output_dir1kminm', type=str)
    parser.add_argument('--log_dir', default='output_dir1kminm', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--save_interval', default=50, type=int, help='Save checkpoint every N epochs')
    return parser


def main(args):
    if misc.is_main_process():
        wandb.init(
            project="MInM",
            entity="visual-intelligence-laboratory",
            config=vars(args),
            name="MinM_imagenet1k_bs256_epoch600_withevaluate_cosine_dataloadfix"
        )

    misc.init_distributed_mode(args)
    print(f"Job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"Args: {args}")

    device = torch.device(args.device)
    torch.manual_seed(args.seed + misc.get_rank())
    np.random.seed(args.seed + misc.get_rank())
    cudnn.benchmark = True

    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_train = MaskedImageDataset(
        image_root=os.path.join(args.data_path, "train"),
        mask_root=os.path.join(args.data_path, "sam/masks_applied"),
        transform=transform_train
    )
    if len(dataset_train) == 0:
        raise ValueError("❌ Dataset is empty! Check data paths.")

    print(f"📌 Training dataset size: {len(dataset_train)}")

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True
    ) if args.distributed else torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )

    # Model setup
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        # 防止 NCCL 报错
        try:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        except Exception as e:
            print("⚠️ DistributedDataParallel failed, fallback to single GPU mode:", e)
            args.distributed = False
            model_without_ddp = model
    else:
        model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    args.lr = args.blr * eff_batch_size / 256 if args.lr is None else args.lr
    print(f"Base LR: {args.lr * 256 / eff_batch_size:.2e}, Actual LR: {args.lr:.2e}")
    print(f"Accumulate grad iterations: {args.accum_iter}, Effective batch size: {eff_batch_size}")

    # Optimizer and scheduler
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Logging setup
    log_writer = SummaryWriter(log_dir=args.log_dir) if misc.is_main_process() and args.log_dir else None
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_acc1 = 0.0  # 跟踪最佳Top-1准确率
    best_acc5 = 0.0  # 跟踪最佳Top-5准确率（如果适用）

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # Warmup learning rate
        if epoch < args.warmup_epochs:
            lr_scale = min(1.0, float(epoch + 1) / args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * args.lr
        elif epoch == args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr  # Reset to full LR before cosine annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)
        else:
            scheduler.step()

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
        )

        current_lr = optimizer.param_groups[0]['lr']
        if misc.is_main_process():
            wandb.log({'epoch': epoch, 'train_loss': train_stats.get('loss', None), 'learning_rate': current_lr})

        # 每5个epoch或最后一个epoch执行Linear Probing
        if (epoch + 1) % 100 == 99 or epoch + 1 == args.epochs:
            # 保存临时模型
            checkpoint_path = os.path.join(args.output_dir, "temporary.pth")
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                name="temporary.pth"
            )

            # 执行Linear Probing，获取Top-1和Top-5准确率
            if misc.is_main_process():
                current_acc1, current_acc5 = linear_probing(checkpoint_path, args)
                print(f"Epoch {epoch+1} - Current best accuracy after linear probing: {{'current_acc1': {current_acc1:.2f}%, 'current_acc5': {current_acc5:.2f}%}}")
                wandb.log({
                    'epoch': epoch,
                    'linear_probing_acc1': current_acc1,
                    'linear_probing_acc5': current_acc5,
                    'best_acc1': best_acc1,
                    'best_acc5': best_acc5
                })

                # 更新最佳模型
                if current_acc1 > best_acc1:
                    best_acc1 = current_acc1
                    best_acc5 = current_acc5  # 同步更新Top-5
                    if args.output_dir:
                        misc.save_model(
                            args=args,
                            model=model,
                            model_without_ddp=model_without_ddp,
                            optimizer=optimizer,
                            loss_scaler=loss_scaler,
                            epoch=epoch,
                            name="best_acc.pth"
                        )

        # Periodic checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
                'loss_scaler': loss_scaler.state_dict()
            }, checkpoint_path, _use_new_zipfile_serialization=False)

        # Logging
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if misc.is_main_process() and args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            if log_writer:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")
    if misc.is_main_process():
        wandb.log({'total_training_time': total_time})
        wandb.finish()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)