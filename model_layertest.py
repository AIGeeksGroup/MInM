import torch
import torch.nn as nn

class ViTBackbone(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=1024, depth=12, num_heads=16, pretrained=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 假设 Patch Embedding 是一个 2D 卷积
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # 加载预训练权重
        if pretrained:
            print(f"🔹 Loading checkpoint from {pretrained}")
            ckpt = torch.load(pretrained, map_location="cpu")
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(ckpt, strict=False)
            print(f"✅ Loaded checkpoint! Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, C, H, W] -> [B, embed_dim, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)  # 变成 [B, N_patches, C]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # 复制 CLS Token
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N_patches+1, C]
        x = x + self.pos_embed  # 位置编码
        x = self.encoder(x)  # Transformer Encoder
        x = self.norm(x)

        print(f"🔹 ViT raw output shape: {x.shape}")  # Debug
        if x.shape[1] == 1:
            print("🚨 ViT 只输出了 CLS Token，特征映射可能有问题！")

        x = x[:, 1:, :]  # 去掉 CLS Token
        H = W = int(x.shape[1] ** 0.5)  # 计算 feature map 高宽
        C = x.shape[-1]
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        print(f"✅ Final feature map shape: {x.shape}")  # Debug
        return x

    def forward(self, x):
        return self.forward_features(x)

# 测试 ViT Backbone
def test_ckpt(ckpt_path):
    print(f"🔹 Testing ViT checkpoint: {ckpt_path}")

    model = ViTBackbone(pretrained=ckpt_path)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)

    print(f"🔹 Output shape from ViT backbone: {out.shape}")
    if out.shape[1:] != (1024, 14, 14):
        print("🚨 WARNING: Output shape does not match expected (B, 1024, 14, 14)!")
    else:
        print("✅ Output shape matches expected format for FPN!")

# 运行测试
test_ckpt("/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/MAE/mae/MinM/output_dir6/vit_mmdet.pth")

