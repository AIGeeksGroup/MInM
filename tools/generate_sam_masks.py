# Step 1: Download imagewoof from HuggingFace
# Step 2: Run SAM instance segmentation -> instance/{wnid}/*_mask.png
# Step 3: Generate sam075/masks_applied via sam_topk_multiple.py
#
# Usage: python tools/generate_sam_masks.py [--sam_checkpoint PATH]

import argparse
import os
import sys
import numpy as np
import cv2
from PIL import Image

import torch

# Monkey-patch torch.as_tensor to handle numpy 2.x dtype incompatibility
_original_as_tensor = torch.as_tensor
def _patched_as_tensor(data, *args, **kwargs):
    if isinstance(data, np.ndarray):
        # Use torch.tensor() which copies data and avoids numpy dtype issues
        device = kwargs.pop('device', None)
        dtype = kwargs.pop('dtype', None)
        t = torch.tensor(data, dtype=dtype)
        if device is not None:
            t = t.to(device=device)
        return t
    return _original_as_tensor(data, *args, **kwargs)
torch.as_tensor = _patched_as_tensor

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tools.sam_topk_multiple import process_and_save_images

# ImageWoof label -> WordNet ID
LABEL_TO_WNID = {
    0: "n02086240",  # Australian terrier
    1: "n02087394",  # Border terrier
    2: "n02088364",  # Samoyed
    3: "n02089973",  # Beagle
    4: "n02093754",  # Shih-Tzu
    5: "n02096294",  # English foxhound
    6: "n02099601",  # Rhodesian ridgeback
    7: "n02105641",  # Dingo
    8: "n02111889",  # Golden retriever
    9: "n02115641",  # Old English sheepdog
}


def download_imagewoof(data_root):
    """Download ImageWoof from HuggingFace, save as ImageNet-style folders."""
    train_dir = os.path.join(data_root, "imagewoof2", "train")
    val_dir = os.path.join(data_root, "imagewoof2", "val")

    if os.path.exists(train_dir) and len(os.listdir(train_dir)) == 10:
        print(f"ImageWoof already exists at {train_dir}, skipping download.")
        return

    print("Downloading ImageWoof from HuggingFace (full_size)...")
    from datasets import load_dataset
    ds = load_dataset("frgfm/imagewoof", "full_size", trust_remote_code=True)

    for split_name, split_dir in [("train", train_dir), ("validation", val_dir)]:
        split = ds[split_name]
        for wnid in LABEL_TO_WNID.values():
            os.makedirs(os.path.join(split_dir, wnid), exist_ok=True)

        counters = {label: 0 for label in LABEL_TO_WNID}
        for sample in split:
            label = sample["label"]
            img = sample["image"]
            wnid = LABEL_TO_WNID[label]
            counters[label] += 1
            fname = f"{wnid}_{counters[label]:05d}.JPEG"
            img.convert("RGB").save(os.path.join(split_dir, wnid, fname), "JPEG")

        total = sum(counters.values())
        print(f"  {split_name}: saved {total} images to {split_dir}")

    print("Download complete.")


def masks_to_instance_rgb(masks):
    """Convert SAM output masks to a single RGB instance segmentation image.

    Each instance gets a unique random color, background is black (0,0,0).
    This matches the format used by sam_topk_multiple.py.
    """
    if len(masks) == 0:
        return None

    h, w = masks[0]["segmentation"].shape
    instance_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Sort by area descending (larger instances first, smaller on top)
    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)

    # Generate unique colors (avoid black)
    np.random.seed(42)
    colors = []
    for _ in range(len(sorted_masks)):
        while True:
            color = tuple(np.random.randint(1, 256, size=3).tolist())
            if color != (0, 0, 0) and color not in colors:
                colors.append(color)
                break

    for mask_data, color in zip(sorted_masks, colors):
        seg = mask_data["segmentation"]  # bool array (H, W)
        instance_img[seg] = color

    return instance_img


def generate_instance_masks(data_root, sam_checkpoint, sam_model_type, device):
    """Run SAM on all train images, save instance segmentation as RGB PNGs."""
    train_dir = os.path.join(data_root, "imagewoof2", "train")
    instance_dir = os.path.join(data_root, "imagewoof2", "instance")

    if not os.path.exists(train_dir):
        raise ValueError(f"Train dir not found: {train_dir}. Run download first.")

    # Check if already done
    done_count = 0
    total_count = 0
    for wnid in sorted(os.listdir(train_dir)):
        wnid_train = os.path.join(train_dir, wnid)
        wnid_inst = os.path.join(instance_dir, wnid)
        if not os.path.isdir(wnid_train):
            continue
        images = [f for f in os.listdir(wnid_train) if f.endswith(".JPEG")]
        total_count += len(images)
        if os.path.exists(wnid_inst):
            done_count += len([f for f in os.listdir(wnid_inst) if f.endswith("_mask.png")])

    if done_count >= total_count and total_count > 0:
        print(f"Instance masks already generated ({done_count}/{total_count}), skipping.")
        return

    print(f"Loading SAM model ({sam_model_type}) from {sam_checkpoint}...")
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    print(f"Generating instance masks for {total_count} images...")
    processed = 0
    skipped = 0

    for wnid in sorted(os.listdir(train_dir)):
        wnid_train = os.path.join(train_dir, wnid)
        wnid_inst = os.path.join(instance_dir, wnid)
        if not os.path.isdir(wnid_train):
            continue

        os.makedirs(wnid_inst, exist_ok=True)
        images = sorted([f for f in os.listdir(wnid_train) if f.endswith(".JPEG")])

        for img_name in images:
            mask_name = img_name.replace(".JPEG", "_mask.png")
            mask_path = os.path.join(wnid_inst, mask_name)

            # Skip if already exists
            if os.path.exists(mask_path):
                skipped += 1
                continue

            img_path = os.path.join(wnid_train, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"  WARNING: cannot read {img_path}, skipping.")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize large images to avoid INT_MAX tensor limit in SAM
            MAX_SIDE = 1500
            h, w = image_rgb.shape[:2]
            if max(h, w) > MAX_SIDE:
                scale = MAX_SIDE / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Ensure contiguous numpy array with explicit dtype (fix torch.as_tensor compatibility)
            image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)

            # Run SAM
            try:
                masks = mask_generator.generate(image_rgb)
            except RuntimeError as e:
                print(f"  WARNING: SAM failed on {img_path}: {e}, skipping.")
                continue

            # Convert to RGB instance map
            instance_img = masks_to_instance_rgb(masks)
            if instance_img is None:
                print(f"  WARNING: no masks for {img_path}, skipping.")
                continue

            # Save as RGB PNG (same format as imagenette instance masks)
            cv2.imwrite(mask_path, cv2.cvtColor(instance_img, cv2.COLOR_RGB2BGR))

            processed += 1
            if (processed + skipped) % 200 == 0:
                print(f"  Progress: {processed + skipped}/{total_count} "
                      f"(processed={processed}, skipped={skipped})")

    print(f"Instance mask generation complete: {processed} new, {skipped} skipped.")


def generate_sam075_masks(data_root):
    """Run sam_topk_multiple.py to generate sam075/masks_applied from instance masks."""
    instance_dir = os.path.join(data_root, "imagewoof2", "instance")
    train_dir = os.path.join(data_root, "imagewoof2", "train")
    target_dir = os.path.join(data_root, "imagewoof2", "sam075")
    masks_applied_dir = os.path.join(target_dir, "masks_applied")

    if os.path.exists(masks_applied_dir) and len(os.listdir(masks_applied_dir)) > 0:
        # Check count
        existing = sum(len(os.listdir(os.path.join(masks_applied_dir, d)))
                       for d in os.listdir(masks_applied_dir)
                       if os.path.isdir(os.path.join(masks_applied_dir, d)))
        if existing > 0:
            print(f"sam075/masks_applied already has {existing} files, skipping.")
            return

    if not os.path.exists(instance_dir):
        raise ValueError(f"Instance dir not found: {instance_dir}")

    print("Generating sam075/masks_applied from instance masks...")
    log_path = os.path.join(data_root, "sam075_log.txt")
    process_and_save_images(
        source_segmentation_folder=instance_dir,
        source_original_folder=train_dir,
        target_folder=target_dir,
        log_filename=log_path,
    )
    print("sam075 mask generation complete.")


def main():
    parser = argparse.ArgumentParser("Generate SAM instance masks for imagewoof")
    parser.add_argument("--data_dir", default="data/imagewoof",
                        help="Data root relative to project root")
    parser.add_argument("--sam_checkpoint", default="sam_vit_h_4b8939.pth",
                        help="Path to SAM checkpoint (absolute or relative to project root)")
    parser.add_argument("--sam_model_type", default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model type")
    parser.add_argument("--device", default="cuda", help="Device for SAM inference")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip HuggingFace download step")
    parser.add_argument("--skip_sam", action="store_true",
                        help="Skip SAM inference (only run sam075 generation)")
    args = parser.parse_args()

    data_root = os.path.join(PROJECT_ROOT, args.data_dir)
    os.makedirs(data_root, exist_ok=True)

    # Resolve SAM checkpoint path
    sam_ckpt = args.sam_checkpoint
    if not os.path.isabs(sam_ckpt):
        sam_ckpt = os.path.join(PROJECT_ROOT, sam_ckpt)

    # Step 1: Download
    if not args.skip_download:
        download_imagewoof(data_root)

    # Step 2: SAM instance segmentation
    if not args.skip_sam:
        if not os.path.exists(sam_ckpt):
            print(f"ERROR: SAM checkpoint not found at {sam_ckpt}")
            print("Download it with:")
            print("  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            sys.exit(1)
        generate_instance_masks(data_root, sam_ckpt, args.sam_model_type, args.device)

    # Step 3: sam075/masks_applied
    generate_sam075_masks(data_root)

    print("\nAll done! Data is ready for training.")
    print(f"  Train images: {data_root}/imagewoof2/train/")
    print(f"  Instance masks: {data_root}/imagewoof2/instance/")
    print(f"  SAM075 masks: {data_root}/imagewoof2/sam075/masks_applied/")


if __name__ == "__main__":
    main()
