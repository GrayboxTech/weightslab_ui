import os
import torch

src_dir = 'tinyimagenet-exp1_ckpt'
dst_dir = 'tinyimagenet-exp1_ckpt_cuda'  

os.makedirs(dst_dir, exist_ok=True)

def to_cuda(obj):
    if torch.is_tensor(obj):
        return obj.cuda()
    elif isinstance(obj, dict):
        return {k: to_cuda(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cuda(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_cuda(v) for v in obj)
    else:
        return obj

for fname in os.listdir(src_dir):
    if fname.startswith('ckpt_'):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        print(f'Processing: {src_path} -> {dst_path}')
        checkpoint = torch.load(src_path, map_location='cpu')
        checkpoint_cuda = to_cuda(checkpoint)
        torch.save(checkpoint_cuda, dst_path)
        print(f'Saved: {dst_path}')
