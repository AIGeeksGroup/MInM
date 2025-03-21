import torch

pretrained_path = "/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/MAE/mae/MinM/output_dir6/vit_mmdet.pth"
# pretrained_path = "/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/MAE/mae/test.pth"
state_dict = torch.load(pretrained_path, map_location='cpu')

# 打印出模型的所有参数名称
print("🔹 预训练权重中的 keys:")
print(state_dict.keys())

print("🔹 state_dict keys:")
print(state_dict['state_dict'].keys())  # 这里打印具体参数名
