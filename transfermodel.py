import torch

# ✅ 强制 weights_only=False，避免 PyTorch 2.6+ 默认问题
ckpt = torch.load(
    '/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/MAE/mae/MinM/output_dir6/best_acc.pth',
    map_location='cpu',
    weights_only=False  # 关键改动
)

# ✅ 只保留 `model` 部分
if 'model' in ckpt:
    model_state_dict = ckpt['model']
else:
    raise ValueError("Checkpoint 文件不包含 'model' 关键字，请检查文件格式！")

# ✅ 另存为新的权重文件
new_ckpt_path = '/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/MAE/mae/MinM/output_dir6/best_acc_weights_only.pth'
torch.save(model_state_dict, new_ckpt_path)

print(f"✅ 转换完成，新权重路径: {new_ckpt_path}")

