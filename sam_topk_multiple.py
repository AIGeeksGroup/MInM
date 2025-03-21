import numpy as np
import cv2
import os
import shutil
from collections import Counter

def load_segmentation_image(image_path):
    """加载分割图像并调整为 224x224，并返回 RGB 格式数据"""
    image = cv2.imread(image_path)  # OpenCV 默认是 BGR
    image = cv2.resize(image, (224, 224))  # 统一缩放到 224x224
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    return image_rgb


def get_instance_areas(segmentation_image):
    """计算每个实例的像素数，不包含黑色部分"""
    unique_colors, counts = np.unique(segmentation_image.reshape(-1, 3), axis=0, return_counts=True)
    total_pixels = segmentation_image.shape[0] * segmentation_image.shape[1]

    black = np.array([0, 0, 0])  # 确保黑色为 RGB 格式
    mask = np.all(unique_colors == black, axis=-1)  # 生成布尔掩码，True 表示黑色

    # 过滤掉黑色
    filtered_colors = unique_colors[~mask]
    filtered_counts = counts[~mask]

    instance_areas = {
        tuple(color): count / total_pixels
        for color, count in zip(filtered_colors, filtered_counts)
    }
    return instance_areas


def select_top_k_instances(instance_areas, threshold=0.75):
    """选择 Top K 个实例，使其像素占比尽可能接近 75%"""
    sorted_instances = sorted(instance_areas.items(), key=lambda x: x[1], reverse=True)
    selected_instances, accumulated_ratio = [], 0

    for color, ratio in sorted_instances:
        if accumulated_ratio + ratio > threshold:
            break
        selected_instances.append(color)
        accumulated_ratio += ratio

    return selected_instances


def mask_patches(segmentation_image, selected_instances, patch_size=16):
    """基于 16x16 Patch 级别生成 mask"""
    h, w, _ = segmentation_image.shape
    h_patches = h // patch_size
    w_patches = w // patch_size
    mask = np.zeros((h, w), dtype=np.uint8)

    black = np.array([0, 0, 0])  # RGB 格式的黑色

    for i in range(0, h_patches * patch_size, patch_size):
        for j in range(0, w_patches * patch_size, patch_size):
            patch = segmentation_image[i:i + patch_size, j:j + patch_size]
            patch_pixels = [tuple(p) for p in patch.reshape(-1, 3) if not np.array_equal(p, black)]
            patch_colors = Counter(patch_pixels)

            # 计算当前 patch 内选中实例的像素比例
            top_instance_pixels = sum(patch_colors[color] for color in patch_colors if color in selected_instances)
            patch_ratio = top_instance_pixels / (patch_size * patch_size) if patch_pixels else 0

            if patch_ratio > 0.5:
                mask[i:i + patch_size, j:j + patch_size] = 255  # 该 patch 设为 mask 区域

    return mask


def apply_mask_to_original(original_image_path, mask, log_file):
    """调整原始图像尺寸为 224x224，并基于 Patch mask 进行遮挡"""
    original_image = cv2.imread(original_image_path)

    if original_image is None:
        log_file.write(f"无法加载原始图像: {original_image_path}\n")
        return None

    original_image = cv2.resize(original_image, (224, 224))
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    if np.all(mask == 0):
        log_file.write(f"Mask 全黑: {original_image_path}\n")
        return None

    original_image[mask == 255] = 0
    return original_image


def process_segmentation(segmentation_path, original_path, patch_size=16, log_file=None):
    segmentation_image = load_segmentation_image(segmentation_path)
    instance_areas = get_instance_areas(segmentation_image)
    selected_instances = select_top_k_instances(instance_areas)

    mask = mask_patches(segmentation_image, selected_instances, patch_size)
    masked_original = apply_mask_to_original(original_path, mask, log_file)

    if masked_original is None:
        return None, None

    mask_output_path = segmentation_path.replace('_mask.png', '_mask_applied.png')
    cv2.imwrite(mask_output_path, mask)

    masked_original_output_path = segmentation_path.replace('_mask.png', '_masked_original.JPEG')
    cv2.imwrite(masked_original_output_path, masked_original)

    return mask_output_path, masked_original_output_path


def process_and_save_images(source_segmentation_folder, source_original_folder, target_folder, patch_size=16,
                            log_filename="log.txt"):
    """ 遍历 `source_segmentation_folder` 并处理所有图像 """

    masks_applied_folder = os.path.join(target_folder, "masks_applied")
    masked_originals_folder = os.path.join(target_folder, "masked_originals")

    os.makedirs(masks_applied_folder, exist_ok=True)
    os.makedirs(masked_originals_folder, exist_ok=True)

    with open(log_filename, "w") as log_file:
        subfolders = [subfolder for subfolder in os.listdir(source_segmentation_folder)
                      if os.path.isdir(os.path.join(source_segmentation_folder, subfolder))]

        for subfolder in subfolders:
            subfolder_path_seg = os.path.join(source_segmentation_folder, subfolder)
            subfolder_path_ori = os.path.join(source_original_folder, subfolder)

            target_mask_applied_subfolder = os.path.join(masks_applied_folder, subfolder)
            target_masked_original_subfolder = os.path.join(masked_originals_folder, subfolder)

            os.makedirs(target_mask_applied_subfolder, exist_ok=True)
            os.makedirs(target_masked_original_subfolder, exist_ok=True)

            for file_name in os.listdir(subfolder_path_seg):
                if file_name.endswith('_mask.png'):
                    segmentation_path = os.path.join(subfolder_path_seg, file_name)
                    original_path = os.path.join(subfolder_path_ori, file_name.replace('_mask.png', '.JPEG'))

                    mask_output_path, masked_original_output_path = process_segmentation(
                        segmentation_path, original_path, patch_size, log_file
                    )

                    if mask_output_path is None or masked_original_output_path is None:
                        log_file.write(f"Skipping {segmentation_path} due to errors.\n")
                        continue

                    mask_output_final = os.path.join(target_mask_applied_subfolder, file_name.replace('_mask.png', '_mask_applied.png'))
                    masked_original_output_final = os.path.join(target_masked_original_subfolder, file_name.replace('_mask.png', '_masked_original.JPEG'))

                    shutil.copy(mask_output_path, mask_output_final)
                    shutil.copy(masked_original_output_path, masked_original_output_final)

                    os.remove(mask_output_path)
                    os.remove(masked_original_output_path)

        print(f"✅ 处理完成，所有文件已保存至 {target_folder}")


if __name__ == "__main__":
    # source_segmentation_folder = './imagenette/imagenette/swap'
    # source_original_folder = './imagenette/imagenette/train'
    # target_folder = './imagenette/imagenette/masked2'
    log_filename = './log.txt'

    source_segmentation_folder = '/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/Sam2/sam2/imagenette/swap'  # 分割图像路径
    source_original_folder = '/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/Sam2/sam2/imagenette/train'  # 原始图像路径
    target_folder = '/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/Sam2/sam2/imagenette/mask'  # 输出路径

    process_and_save_images(source_segmentation_folder, source_original_folder, target_folder,
                            log_filename=log_filename)
