import argparse
import torch
from collections import OrderedDict

def vit_to_mmdet(src, dst):
    """
    Convert ViT pretrained weights to MMDetection-compatible format.
    """
    # Load the pretrained ViT model
    vit_model = torch.load(src, map_location='cpu')
    state_dict = vit_model['state_dict'] if 'state_dict' in vit_model else vit_model
    
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        # Ignore decoder layers
        if 'decoder' in k:
            continue
        
        # Remove CLS token if present
        if 'cls_token' in k:
            continue
        
        # Rename patch embedding layer
        if k.startswith('patch_embed.'):
            new_k = k.replace('patch_embed.', 'backbone.patch_embed.')
        
        # Rename transformer blocks
        elif k.startswith('blocks.'):
            new_k = k.replace('blocks.', 'backbone.layers.')
        
        # Rename norm layers
        elif k.startswith('norm.'):
            new_k = k.replace('norm.', 'backbone.norm.')
        
        else:
            new_k = f'backbone.{k}'
        
        new_state_dict[new_k] = v
    
    # Save converted weights
    converted_ckpt = {'state_dict': new_state_dict}
    torch.save(converted_ckpt, dst)
    print(f"✅ Converted model saved to: {dst}")

def main():
    parser = argparse.ArgumentParser(description='Convert ViT weights to MMDetection format')
    parser.add_argument('src', help='Path to source ViT checkpoint')
    parser.add_argument('dst', help='Path to save converted checkpoint')
    args = parser.parse_args()
    
    vit_to_mmdet(args.src, args.dst)

if __name__ == '__main__':
    main()

