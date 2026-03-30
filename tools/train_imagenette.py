# MInM training on imagenette with small-batch validation
# Data source: data/imagenette/imagenette.tar.gz (train + SAM masks)
# Probing data: data/imagenette/imagenette_probing.tar.gz
# Usage: python tools/train_imagenette.py [--epochs N] [--batch_size N]

import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import tarfile
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
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models import models_minm as models_mae
from engine.engine_pretrain_minm import train_one_epoch
from engine.engine_probing_minm import linear_probing
import wandb

wandb.login(key="9c1af0a383f6b1138e4ab20f65a4d6e4f194ffad")

from models.models_minm import InstanceGuidedMasking


def extract_tar_if_needed(tar_path, extract_dir):
    """Extract tar.gz if not already extracted."""
    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
        print(f"Already extracted at {extract_dir}, skipping.")
        return
    print(f"Extracting {tar_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
    print(f"Extraction complete.")


class MaskedImageDataset(Dataset):
    """Dataset for MInM: loads images + SAM masks with synchronized augmentation.
    Adapted from imagenet_1kminm_parallel.py for imagenette.
    """
    def __init__(self, image_root, mask_root, patch_size=16, img_size=224):
        self.image_paths = []
        self.mask_paths = []
        self.class_indices = []
        self.processor = InstanceGuidedMasking(patch_size, img_size)

        class_folders = sorted(os.listdir(image_root))
        for class_idx, class_name in enumerate(class_folders):
            image_class_dir = os.path.join(image_root, class_name)
            mask_class_dir = os.path.join(mask_root, class_name)

            if not os.path.isdir(image_class_dir) or not os.path.isdir(mask_class_dir):
                continue

            images = sorted([f for f in os.listdir(image_class_dir) if f.endswith(".JPEG")])
            for img_name in images:
                image_path = os.path.join(image_class_dir, img_name)
                mask_name = img_name.replace(".JPEG", "_mask_applied.png")
                mask_path = os.path.join(mask_class_dir, mask_name)

                if os.path.exists(mask_path):
                    self.image_paths.append(image_path)
                    self.mask_paths.append(mask_path)
                    self.class_indices.append(class_idx)

        print(f"Loaded {len(self.image_paths)} images with masks from {image_root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # Resize to match mask resolution
        image = TF.resize(image, [self.processor.img_size, self.processor.img_size],
                          interpolation=TF.InterpolationMode.BICUBIC)

        # Synchronized spatial transforms for image and mask
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.2, 1.0), ratio=(3./4., 4./3.))
        image = TF.resized_crop(image, i, j, h, w,
                                [self.processor.img_size, self.processor.img_size],
                                interpolation=TF.InterpolationMode.BICUBIC)
        mask = TF.resized_crop(mask, i, j, h, w,
                               [self.processor.img_size, self.processor.img_size],
                               interpolation=TF.InterpolationMode.NEAREST)

        if torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Color augmentations (image only, no effect on mask)
        image = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)(image)
        if torch.rand(1).item() > 0.8:
            image = TF.to_grayscale(image, num_output_channels=3)
        if torch.rand(1).item() > 0.5:
            image = TF.gaussian_blur(image, kernel_size=23, sigma=(0.1, 2.0))

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = TF.to_tensor(mask).squeeze(0)
        processed_mask = self.processor._process_mask(mask)

        return image, processed_mask, self.class_indices[idx]


def get_args_parser():
    parser = argparse.ArgumentParser('MInM bug-fix test on imagenette', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=True)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    # Data paths (relative to project root)
    parser.add_argument('--data_tar', default='data/imagenette/imagenette.tar.gz', type=str)
    parser.add_argument('--probing_tar', default='data/imagenette/imagenette_probing.tar.gz', type=str)
    parser.add_argument('--data_extract_dir', default='data/imagenette/extracted', type=str)
    parser.add_argument('--probing_extract_dir', default='data/imagenette/probing_extracted', type=str)
    parser.add_argument('--output_dir', default='output_bugfix_imagenette', type=str)
    parser.add_argument('--log_dir', default='output_bugfix_imagenette', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--save_interval', default=5, type=int)
    # imagenette has 10 classes
    parser.add_argument('--nb_classes', default=10, type=int)
    return parser


def main(args):
    if misc.is_main_process():
        wandb.init(
            project="MInM",
            entity="visual-intelligence-laboratory",
            config=vars(args),
            name="MInM_imagenette_mr075_blr1e3_wd005_ep100"
        )

    misc.init_distributed_mode(args)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"Args: {args}")

    device = torch.device(args.device)
    torch.manual_seed(args.seed + misc.get_rank())
    np.random.seed(args.seed + misc.get_rank())
    cudnn.benchmark = True

    # Extract data
    data_tar = os.path.join(project_root, args.data_tar)
    probing_tar = os.path.join(project_root, args.probing_tar)
    data_extract = os.path.join(project_root, args.data_extract_dir)
    probing_extract = os.path.join(project_root, args.probing_extract_dir)

    extract_tar_if_needed(data_tar, data_extract)
    extract_tar_if_needed(probing_tar, probing_extract)

    # Paths inside extracted data
    # imagenette2/train, imagenette2/sam075/masks_applied
    image_root = os.path.join(data_extract, "imagenette2", "train")
    mask_root = os.path.join(data_extract, "imagenette2", "sam075", "masks_applied")
    probing_path = os.path.join(data_extract, "imagenette2")

    dataset_train = MaskedImageDataset(
        image_root=image_root,
        mask_root=mask_root,
    )
    if len(dataset_train) == 0:
        raise ValueError("Dataset is empty! Check data paths.")

    print(f"Training dataset size: {len(dataset_train)}")

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True
    ) if args.distributed else torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )

    # Model setup
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, use_instance_mask=True)
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        try:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        except Exception as e:
            print(f"DistributedDataParallel failed, fallback to single GPU: {e}")
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # Logging setup
    log_writer = SummaryWriter(log_dir=args.log_dir) if misc.is_main_process() and args.log_dir else None
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_acc1 = 0.0
    best_acc5 = 0.0

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
                pg['lr'] = args.lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)
        else:
            scheduler.step()

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args
        )

        current_lr = optimizer.param_groups[0]['lr']
        if misc.is_main_process():
            wandb.log({'epoch': epoch, 'train_loss': train_stats.get('loss', None),
                       'learning_rate': current_lr})

        # Linear probing every 5 epochs or at last epoch
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, "temporary.pth")
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler,
                epoch=epoch, name="temporary.pth"
            )

            if misc.is_main_process():
                sys.argv_backup = sys.argv
                sys.argv = [sys.argv[0],
                            '--data_path', probing_path,
                            '--nb_classes', str(args.nb_classes),
                            '--epochs', '50',
                            '--batch_size', '64',
                            '--num_workers', '4']
                try:
                    current_acc1, current_acc5 = linear_probing(checkpoint_path, args)
                    print(f"Epoch {epoch+1} - Linear probing: acc1={current_acc1:.2f}%, acc5={current_acc5:.2f}%")
                    wandb.log({
                        'epoch': epoch,
                        'linear_probing_acc1': current_acc1,
                        'linear_probing_acc5': current_acc5,
                        'best_acc1': best_acc1,
                        'best_acc5': best_acc5
                    })

                    if current_acc1 > best_acc1:
                        best_acc1 = current_acc1
                        best_acc5 = current_acc5
                        if args.output_dir:
                            misc.save_model(
                                args=args, model=model,
                                model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler,
                                epoch=epoch, name="best_acc.pth"
                            )
                except Exception as e:
                    print(f"Linear probing failed at epoch {epoch+1}: {e}")
                finally:
                    sys.argv = sys.argv_backup

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
