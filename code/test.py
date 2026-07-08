import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm

from model import Network
from tools import MemoryFriendlyLoader, save_images

ARCH_KEYS = [
    'stage',
    'embed_dim',
    'patch_size',
    'illu_min',
    'retinex_eps',
    'stage_update_max',
    'stage_positive_scale',
    'stage_negative_scale',
    'final_denoise_hidden',
    'final_denoise_max_residual',
]

def parse_args():
    parser = argparse.ArgumentParser('SCI-Mamba inference')
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/data/test')
    parser.add_argument('--save_path', type=str, default='./result')
    parser.add_argument('--model_weights', type=str, default='./mamba_sci/Mamba-SCI-20260615-185807/weights/mamba_sci_final.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--max_size', type=int, default=512)
    parser.add_argument('--no_resize', action='store_true')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--stage', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--illu_min', type=float, default=0.05)
    parser.add_argument('--retinex_eps', type=float, default=1e-3)
    parser.add_argument('--stage_update_max', type=float, default=0.20)
    parser.add_argument('--stage_positive_scale', type=float, default=0.70)
    parser.add_argument('--stage_negative_scale', type=float, default=0.90)
    parser.add_argument('--final_denoise_hidden', type=int, default=16)
    parser.add_argument('--final_denoise_max_residual', type=float, default=0.04)
    parser.add_argument('--strict_load', action='store_true')
    parser.add_argument('--ignore_checkpoint_args', action='store_true')
    parser.add_argument('--ref_mode', type=str, default='highres_retinex', choices=['upsample_ref', 'highres_retinex', 'highres_retinex_denoise'])
    parser.add_argument('--illu_upsample', type=str, default='safe_bilinear', choices=['safe_bilinear', 'bilinear', 'guided'])
    parser.add_argument('--guided_radius', type=int, default=8)
    parser.add_argument('--guided_eps', type=float, default=1e-3)
    parser.add_argument('--guided_blend', type=float, default=0.35)
    parser.add_argument('--illu_smooth_radius', type=int, default=1)
    parser.add_argument('--illu_smooth_blend', type=float, default=0.18)
    return parser.parse_args()

def apply_checkpoint_args(args, checkpoint):
    if args.ignore_checkpoint_args or not isinstance(checkpoint, dict):
        return
    saved_args = checkpoint.get('args')
    if not isinstance(saved_args, dict):
        return
    for key in ARCH_KEYS:
        if key in saved_args:
            value = saved_args[key]
            current = getattr(args, key)
            try:
                if isinstance(current, bool):
                    value = bool(value)
                elif isinstance(current, int) and not isinstance(current, bool):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
            except (TypeError, ValueError):
                continue
            setattr(args, key, value)

def build_model(args):
    return Network(
        stage=args.stage,
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        illu_min=args.illu_min,
        retinex_eps=args.retinex_eps,
        stage_update_max=args.stage_update_max,
        stage_positive_scale=args.stage_positive_scale,
        stage_negative_scale=args.stage_negative_scale,
        final_denoise_hidden=args.final_denoise_hidden,
        final_denoise_max_residual=args.final_denoise_max_residual,
    )

def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.endswith('total_ops') or key.endswith('total_params'):
            continue
        if key.startswith('module.'):
            key = key[len('module.'):]
        cleaned[key] = value
    return cleaned

def load_model(args, device):
    checkpoint = torch.load(args.model_weights, map_location=device)
    apply_checkpoint_args(args, checkpoint)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = clean_state_dict(state_dict)
    model = build_model(args)
    if args.strict_load:
        model.load_state_dict(state_dict, strict=True)
    else:
        model_state = model.state_dict()
        compatible = {key: value for key, value in state_dict.items() if key in model_state and model_state[key].shape == value.shape}
        model.load_state_dict(compatible, strict=False)
    model.to(device)
    model.eval()
    return model

def pad_to_patch_size(x, patch_size):
    h, w = int(x.shape[-2]), int(x.shape[-1])
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    mode = 'reflect' if h > pad_h and w > pad_w else 'replicate'
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), pad_h, pad_w

def resize_for_inference(x, max_size, no_resize):
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if no_resize or max(h, w) <= max_size:
        return x
    scale = float(max_size) / float(max(h, w))
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    return F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)

def box_mean(x, radius):
    radius = int(radius)
    if radius <= 0:
        return x
    h, w = int(x.shape[-2]), int(x.shape[-1])
    mode = 'reflect' if h > radius and w > radius else 'replicate'
    x = F.pad(x, (radius, radius, radius, radius), mode=mode)
    return F.avg_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=0)

def rgb_to_gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

def guided_filter_gray(guidance_gray, source, radius, eps):
    radius = int(radius)
    if radius <= 0:
        return source
    guidance = guidance_gray.float()
    source = source.float()
    mean_i = box_mean(guidance, radius)
    mean_p = box_mean(source, radius)
    corr_i = box_mean(guidance * guidance, radius)
    corr_ip = box_mean(guidance * source, radius)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(eps))
    b = mean_p - a * mean_i
    return box_mean(a, radius) * guidance + box_mean(b, radius)

def upsample_illumination(args, input_high, illu_low, target_size):
    same_size = tuple(illu_low.shape[-2:]) == tuple(target_size)
    if same_size:
        illu_up = illu_low
    else:
        illu_up = F.interpolate(illu_low, size=target_size, mode='bilinear', align_corners=False)
    if args.illu_upsample == 'guided' and not same_size:
        blend = max(0.0, min(1.0, float(args.guided_blend)))
        if blend > 0:
            guided = guided_filter_gray(rgb_to_gray(input_high.float()), illu_up.float(), args.guided_radius, args.guided_eps)
            illu_up = (1.0 - blend) * illu_up.float() + blend * guided.float()
    elif args.illu_upsample == 'safe_bilinear':
        blend = max(0.0, min(1.0, float(args.illu_smooth_blend)))
        if args.illu_smooth_radius > 0 and blend > 0:
            smooth = box_mean(illu_up.float(), args.illu_smooth_radius)
            illu_up = (1.0 - blend) * illu_up.float() + blend * smooth
    return torch.clamp(illu_up, args.illu_min, 1.0).to(dtype=illu_low.dtype)

def output_path_for(args, image_path):
    rel_path = os.path.relpath(image_path, args.data_path)
    rel_stem = os.path.splitext(rel_path)[0] + '.png'
    return os.path.join(args.save_path, 'images', rel_stem)

def run_inference(args, model, input_high):
    orig_h, orig_w = int(input_high.shape[-2]), int(input_high.shape[-1])
    input_low = resize_for_inference(input_high, args.max_size, args.no_resize)
    input_low, pad_h, pad_w = pad_to_patch_size(input_low, model.patch_size)
    if args.ref_mode == 'upsample_ref':
        i_list, r_list, _, _, _, _ = model(input_low)
        illu_low = torch.clamp(i_list[-1], args.illu_min, 1.0)
        ref_low = torch.clamp(r_list[-1], 0.0, 1.0)
    else:
        illu_low = torch.clamp(model.forward_final_illumination(input_low), args.illu_min, 1.0)
        ref_low = None
    valid_h = int(input_low.shape[-2]) - pad_h
    valid_w = int(input_low.shape[-1]) - pad_w
    illu_low = illu_low[:, :, :valid_h, :valid_w]
    if ref_low is not None:
        ref_low = ref_low[:, :, :valid_h, :valid_w]
    illu_high = upsample_illumination(args, input_high, illu_low, target_size=(orig_h, orig_w))
    if args.ref_mode == 'upsample_ref':
        if tuple(ref_low.shape[-2:]) != (orig_h, orig_w):
            ref_high = F.interpolate(ref_low, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        else:
            ref_high = ref_low
    else:
        ref_high = torch.clamp(input_high.float() / (illu_high.float() + args.retinex_eps), 0.0, 1.0)
        if args.ref_mode == 'highres_retinex_denoise':
            ref_high, _ = model.final_ref_denoiser(input_high, illu_high, ref_high)
    return torch.clamp(ref_high, 0.0, 1.0)

def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for inference.')
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    model = load_model(args, device)
    dummy = torch.randn(1, 3, args.max_size, args.max_size, device=device)
    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            if args.ref_mode == 'upsample_ref':
                _ = model(dummy)
            else:
                _ = model.forward_final_illumination(dummy)
    torch.cuda.synchronize()

    dataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')
    if len(dataset) == 0:
        raise RuntimeError('No supported images were found.')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

    with torch.inference_mode():
        for input_high, image_path_list in tqdm(loader, desc='Testing', dynamic_ncols=True):
            input_high = input_high.to(device, non_blocking=True)
            image_path = image_path_list[0] if isinstance(image_path_list, (list, tuple)) else image_path_list
            result = run_inference(args, model, input_high)
            save_images(result, output_path_for(args, image_path))

if __name__ == '__main__':
    main()
