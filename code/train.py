import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from loss import LossFunction
from model import Network
from tools import AvgrageMeter, MemoryFriendlyLoader

def parse_args():
    parser = argparse.ArgumentParser('SCI-Mamba training')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum_steps', type=int, default=2)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--stage', type=int, default=3)
    parser.add_argument('--stage_weights', type=float, nargs='+', default=[0.50, 0.75, 1.00])
    parser.add_argument('--save', type=str, default='mamba_sci')
    parser.add_argument('--train_dir', type=str, default='/root/autodl-tmp/data/train')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--illu_min', type=float, default=0.05)
    parser.add_argument('--retinex_eps', type=float, default=1e-3)
    parser.add_argument('--stage_update_max', type=float, default=0.20)
    parser.add_argument('--stage_positive_scale', type=float, default=0.70)
    parser.add_argument('--stage_negative_scale', type=float, default=0.90)
    parser.add_argument('--final_denoise_hidden', type=int, default=16)
    parser.add_argument('--final_denoise_max_residual', type=float, default=0.04)
    parser.add_argument('--w_prior', type=float, default=0.006)
    parser.add_argument('--w_smooth', type=float, default=0.003)
    parser.add_argument('--w_illu_color', type=float, default=0.006)
    parser.add_argument('--w_ref_color', type=float, default=0.25)
    parser.add_argument('--w_target', type=float, default=2.2)
    parser.add_argument('--ref_color_ratio_min', type=float, default=0.50)
    parser.add_argument('--ref_color_ratio_max', type=float, default=1.50)
    parser.add_argument('--ref_color_spread_weight', type=float, default=0.02)
    parser.add_argument('--ref_color_extreme_weight', type=float, default=1.00)
    parser.add_argument('--target_median_kernel', type=int, default=3)
    parser.add_argument('--target_gaussian_sigma', type=float, default=0.6)
    parser.add_argument('--target_bright_otsu_scale', type=float, default=0.78)
    parser.add_argument('--target_bright_soft_high_percentile', type=float, default=99.5)
    parser.add_argument('--target_bright_soft_blur_sigma', type=float, default=0.8)
    parser.add_argument('--target_texture_otsu_scale', type=float, default=0.82)
    parser.add_argument('--target_edge_low_percentile', type=float, default=85.0)
    parser.add_argument('--target_edge_high_percentile', type=float, default=99.0)
    parser.add_argument('--target_edge_soft_blur_sigma', type=float, default=0.6)
    parser.add_argument('--target_seed_dilate_radius', type=int, default=2)
    parser.add_argument('--target_seed_close_radius', type=int, default=3)
    parser.add_argument('--target_fill_dilate_radius', type=int, default=1)
    parser.add_argument('--target_min_area', type=int, default=300)
    parser.add_argument('--target_feature_region_blur_sigma', type=float, default=1.2)
    parser.add_argument('--target_feature_region_base', type=float, default=0.45)
    parser.add_argument('--target_feature_edge_gain', type=float, default=0.55)
    parser.add_argument('--target_strength_blur_sigma', type=float, default=0.8)
    parser.add_argument('--target_illu_min', type=float, default=0.10)
    parser.add_argument('--target_illu_max', type=float, default=1.00)
    parser.add_argument('--target_dark_feature_weight', type=float, default=1.35)
    parser.add_argument('--target_bright_feature_weight', type=float, default=0.12)
    parser.add_argument('--target_background_pull', type=float, default=0.90)
    parser.add_argument('--target_dark_region_weight', type=float, default=2.10)
    parser.add_argument('--target_bright_region_weight', type=float, default=0.25)
    parser.add_argument('--target_background_region_weight', type=float, default=0.45)
    parser.add_argument('--target_loss_type', type=str, default='charbonnier', choices=['charbonnier', 'l1', 'mae'])
    parser.add_argument('--target_final_scale', type=float, default=1.00)
    parser.add_argument('--target_early_scale', type=float, default=0.10)
    parser.add_argument('--target_middle_scale', type=float, default=0.35)
    parser.add_argument('--target_stage_mode', type=str, default='all', choices=['early_only', 'final_only', 'early_final', 'all'])
    parser.add_argument('--final_size', type=int, nargs='+', default=[256])
    parser.add_argument('--crop_sizes', type=int, nargs='+', default=[256, 288])
    parser.add_argument('--photometric_aug', action='store_true')
    parser.add_argument('--vflip_aug', action='store_true')
    parser.add_argument('--rotate_aug', action='store_true')
    parser.add_argument('--no_amp', action='store_true')
    return parser.parse_args()

def validate_args(args):
    if args.stage <= 0:
        raise ValueError('--stage must be positive.')
    if args.batch_size <= 0:
        raise ValueError('--batch_size must be positive.')
    if args.grad_accum_steps <= 0:
        raise ValueError('--grad_accum_steps must be positive.')
    if args.epochs <= 0:
        raise ValueError('--epochs must be positive.')
    if len(args.stage_weights) != args.stage:
        args.stage_weights = (args.stage_weights * args.stage)[:args.stage]
    if isinstance(args.final_size, list):
        args.final_size = int(args.final_size[0])

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def stage_mode_for_index(index, num_stages):
    if index == 0:
        return 'early'
    if index == num_stages - 1:
        return 'final'
    return 'middle'

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

def build_loss(args):
    return LossFunction(
        w_prior=args.w_prior,
        w_smooth=args.w_smooth,
        w_illu_color=args.w_illu_color,
        w_ref_color=args.w_ref_color,
        w_target=args.w_target,
        eps=args.retinex_eps,
        illu_min=args.illu_min,
        target_median_kernel=args.target_median_kernel,
        target_gaussian_sigma=args.target_gaussian_sigma,
        target_bright_otsu_scale=args.target_bright_otsu_scale,
        target_bright_soft_high_percentile=args.target_bright_soft_high_percentile,
        target_bright_soft_blur_sigma=args.target_bright_soft_blur_sigma,
        target_texture_otsu_scale=args.target_texture_otsu_scale,
        target_edge_low_percentile=args.target_edge_low_percentile,
        target_edge_high_percentile=args.target_edge_high_percentile,
        target_edge_soft_blur_sigma=args.target_edge_soft_blur_sigma,
        target_seed_dilate_radius=args.target_seed_dilate_radius,
        target_seed_close_radius=args.target_seed_close_radius,
        target_fill_dilate_radius=args.target_fill_dilate_radius,
        target_min_area=args.target_min_area,
        target_feature_region_blur_sigma=args.target_feature_region_blur_sigma,
        target_feature_region_base=args.target_feature_region_base,
        target_feature_edge_gain=args.target_feature_edge_gain,
        target_strength_blur_sigma=args.target_strength_blur_sigma,
        target_illu_min=args.target_illu_min,
        target_illu_max=args.target_illu_max,
        target_dark_feature_weight=args.target_dark_feature_weight,
        target_bright_feature_weight=args.target_bright_feature_weight,
        target_background_pull=args.target_background_pull,
        target_dark_region_weight=args.target_dark_region_weight,
        target_bright_region_weight=args.target_bright_region_weight,
        target_background_region_weight=args.target_background_region_weight,
        target_loss_type=args.target_loss_type,
        target_final_scale=args.target_final_scale,
        target_early_scale=args.target_early_scale,
        target_middle_scale=args.target_middle_scale,
        target_stage_mode=args.target_stage_mode,
        ref_color_ratio_min=args.ref_color_ratio_min,
        ref_color_ratio_max=args.ref_color_ratio_max,
        ref_color_spread_weight=args.ref_color_spread_weight,
        ref_color_extreme_weight=args.ref_color_extreme_weight,
    )

def build_scheduler(optimizer, args):
    if args.warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=5e-6)
    warmup = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=max(args.warmup_epochs, 1))
    cosine = CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=5e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

def is_finite_tensor_list(items):
    for item in items:
        if not torch.isfinite(item).all():
            return False
    return True

def save_final_checkpoint(path, epoch, model, optimizer, scheduler, scaler, loss_value, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'loss': loss_value,
            'args': vars(args),
        },
        path,
    )

def train_one_epoch(epoch, model, criterion, optimizer, scaler, train_loader, args, amp_enabled, amp_dtype):
    model.train()
    loss_meter = AvgrageMeter()
    stage_meters = [AvgrageMeter() for _ in range(args.stage)]
    optimizer.zero_grad(set_to_none=True)
    accumulated_steps = 0
    progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}', dynamic_ncols=True, leave=True)

    for batch_idx, (input_im, _) in enumerate(progress):
        input_im = input_im.cuda(non_blocking=True)
        current_loss = None
        stage_values = None

        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            i_list, r_list, _, _, _, _ = model(input_im)
            if len(i_list) == args.stage and len(r_list) == args.stage and is_finite_tensor_list(i_list + r_list):
                shared_guidance = criterion._safe_guidance(input_im)
                total_loss = torch.zeros((), device=input_im.device, dtype=torch.float32)
                values = []
                for stage_index in range(args.stage):
                    illu = torch.clamp(i_list[stage_index], args.illu_min, 1.0)
                    enhanced = torch.clamp(r_list[stage_index], 0.0, 1.0)
                    stage_loss, _, _ = criterion(
                        input_im,
                        illu,
                        enhanced=enhanced,
                        stage_mode=stage_mode_for_index(stage_index, args.stage),
                        guidance=shared_guidance,
                    )
                    if not torch.isfinite(stage_loss):
                        values = None
                        break
                    total_loss = total_loss + args.stage_weights[stage_index] * stage_loss
                    values.append(stage_loss.detach())
                if values is not None and len(values) == args.stage and torch.isfinite(total_loss):
                    current_loss = total_loss
                    stage_values = values

        if current_loss is None:
            optimizer.zero_grad(set_to_none=True)
            accumulated_steps = 0
            progress.set_postfix(loss=f'{loss_meter.avg:.6f}', lr=f'{optimizer.param_groups[0]["lr"]:.2e}')
            continue

        loss_for_backward = current_loss / args.grad_accum_steps
        if scaler is not None:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()
        accumulated_steps += 1

        loss_meter.update(float(current_loss.detach().item()), int(input_im.size(0)))
        for meter, value in zip(stage_meters, stage_values):
            meter.update(float(value.item()), int(input_im.size(0)))

        is_update_step = accumulated_steps >= args.grad_accum_steps or batch_idx + 1 == len(train_loader)
        if is_update_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            if math.isfinite(float(grad_norm)):
                if scaler is not None:
                    scaler.step(optimizer)
                else:
                    optimizer.step()
            if scaler is not None:
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulated_steps = 0

        postfix = {'loss': f'{loss_meter.avg:.6f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'}
        for idx, meter in enumerate(stage_meters):
            postfix[f's{idx + 1}'] = f'{meter.avg:.6f}'
        progress.set_postfix(postfix)

    return loss_meter.avg

def main():
    args = parse_args()
    validate_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training.')
    seed_everything(args.seed)
    cudnn.benchmark = True

    run_dir = os.path.join(args.save, 'Mamba-SCI-{}'.format(time.strftime('%Y%m%d-%H%M%S')))
    weight_dir = os.path.join(run_dir, 'weights')
    final_path = os.path.join(weight_dir, 'mamba_sci_final.pth')

    model = build_model(args).cuda()
    criterion = build_loss(args).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)

    bf16_supported = bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
    amp_enabled = not args.no_amp
    amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
    scaler = None if (not amp_enabled or bf16_supported) else GradScaler(init_scale=512.0, growth_interval=500, growth_factor=1.5)

    train_dataset = MemoryFriendlyLoader(
        img_dir=args.train_dir,
        task='train',
        final_size=args.final_size,
        crop_sizes=args.crop_sizes,
        photometric_aug=args.photometric_aug,
        hflip_aug=True,
        vflip_aug=args.vflip_aug,
        rotate_aug=args.rotate_aug,
    )
    if len(train_dataset) == 0:
        raise RuntimeError('No training images were found.')

    loader_kwargs = {
        'dataset': train_dataset,
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': args.num_workers,
        'shuffle': True,
        'drop_last': True,
    }
    if args.num_workers > 0:
        loader_kwargs.update({'persistent_workers': True, 'prefetch_factor': 4})
    train_loader = torch.utils.data.DataLoader(**loader_kwargs)
    if len(train_loader) == 0:
        raise RuntimeError('The training DataLoader has zero batches.')

    last_loss = math.nan
    for epoch in range(args.epochs):
        last_loss = train_one_epoch(epoch, model, criterion, optimizer, scaler, train_loader, args, amp_enabled, amp_dtype)
        scheduler.step()

    save_final_checkpoint(final_path, args.epochs - 1, model, optimizer, scheduler, scaler, last_loss, args)

if __name__ == '__main__':
    main()
