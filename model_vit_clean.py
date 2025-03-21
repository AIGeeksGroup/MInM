import torch

pretrained_path = "/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/MAE/mae/MinM/output_dir6/best_acc_weights_only.pth"
state_dict = torch.load(pretrained_path, map_location='cpu')

# 只保留 backbone 相关的参数
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder")}

# 打印剩余的 keys，检查是否只包含 backbone
print("🔹 过滤后的预训练权重 keys:")
print(filtered_state_dict.keys())

# 保存清理后的权重，以便 ViT Backbone 使用
torch.save(filtered_state_dict, "/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/MAE/cleaned_vit_backbone.pth")

