import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

def safe_interpolate(x, size=None, scale_factor=None, mode='bilinear', align_corners=False):
    if mode in ('bilinear', 'bicubic', 'trilinear'):
        orig_dtype = x.dtype
        y = F.interpolate(
            x.float(),
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
        return y.to(orig_dtype)
    return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)

def cross_scan_1d(x, h_p, w_p):
    B, D, L = x.shape
    device = x.device
    xs = x.new_empty(B, 4, D, L)
    xs[:, 0] = x

    idx_col = torch.arange(L, device=device).view(h_p, w_p).t().contiguous().view(-1)
    xs[:, 1] = x[:, :, idx_col]
    xs[:, 2] = xs[:, 0].flip(-1)
    xs[:, 3] = xs[:, 1].flip(-1)
    return xs

def cross_merge_1d(ys, h_p, w_p):
    B, K, D, L = ys.shape
    device = ys.device

    idx_col = torch.arange(L, device=device).view(h_p, w_p).t().contiguous().view(-1)
    idx_col_inv = torch.empty_like(idx_col)
    idx_col_inv[idx_col] = torch.arange(L, device=device)

    y_row = ys[:, 0] + ys[:, 2].flip(-1)
    y_col = ys[:, 1] + ys[:, 3].flip(-1)

    y_col = y_col[:, :, idx_col_inv]

    return 0.25 * (y_row + y_col)

class PatchEmbed(nn.Module):

    def __init__(
        self,
        in_channels=3,embed_dim=64,patch_size=4,pos_init_scale=0.05,pos_max_scale=0.20,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=7, stride=patch_size, padding=0)
        self.norm = nn.LayerNorm(embed_dim)

        self.base_h = 256 // patch_size
        self.base_w = 256 // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.base_h, self.base_w))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_max_scale = float(pos_max_scale)
        init_ratio = float(pos_init_scale) / max(self.pos_max_scale, 1e-8)
        init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)
        self.pos_scale_logit = nn.Parameter(torch.logit(torch.tensor(init_ratio, dtype=torch.float32)))

    def forward(self, x):

        pad = 3
        mode = 'reflect' if x.shape[-2] > pad and x.shape[-1] > pad else 'replicate'
        x = F.pad(x, (pad, pad, pad, pad), mode=mode)

        x = self.proj(x)
        h_p, w_p = x.shape[2], x.shape[3]

        if h_p != self.base_h or w_p != self.base_w:
            pos = safe_interpolate(self.pos_embed, size=(h_p, w_p), mode='bicubic')
        else:
            pos = self.pos_embed

        pos_scale = self.pos_max_scale * torch.sigmoid(self.pos_scale_logit)
        x = x + pos_scale.to(dtype=x.dtype, device=x.device) * pos

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h_p, w_p

class SS1D(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.k_group = 4

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.constant_(self.dt_proj.bias, -3.0)

        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1))
        self.A_log = nn.Parameter(A_log)

        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)

    def forward(self, x, h_p, w_p):
        B, N, D = x.shape

        xz = self.in_proj(x)
        x_mamba, gate = xz.chunk(2, dim=-1)

        x_mamba = x_mamba.transpose(1, 2)
        x_mamba = self.act(self.conv1d(x_mamba))

        xs = cross_scan_1d(x_mamba, h_p, w_p)

        L = N
        u = xs.reshape(B * self.k_group, -1, L)
        orig_dtype = u.dtype
        with torch.cuda.amp.autocast(enabled=False):
            u = u.float()

            x_dbl = self.x_proj(u.transpose(1, 2))
            dts, Bs, Cs = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

            delta = self.dt_proj(dts).transpose(1, 2).contiguous()
            delta = torch.clamp(F.softplus(delta), min=1e-5, max=0.1)

            Bs = Bs.transpose(1, 2).unsqueeze(1).contiguous()
            Cs = Cs.transpose(1, 2).unsqueeze(1).contiguous()
            A = -torch.exp(self.A_log.float()).clamp(max=20.0)
            D_param = self.D.float()

            if selective_scan_fn is not None and x.is_cuda:
                ys = selective_scan_fn(
                    u.contiguous(),
                    delta.contiguous(),
                    A,
                    Bs.contiguous(),
                    Cs.contiguous(),
                    D_param,
                    z=None,
                    return_last_state=False,
                )
                if torch.isnan(ys).any() or torch.isinf(ys).any():
                    ys = u.view(B, self.k_group, -1, L)
                else:
                    ys = ys.view(B, self.k_group, -1, L)
            else:
                ys = xs.float()

            y = cross_merge_1d(ys, h_p, w_p)

        y = y.to(orig_dtype)
        gate = self.act(gate.transpose(1, 2))
        out = self.out_proj((y * gate).transpose(1, 2))
        return out

class VSSBlock1D(nn.Module):

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss1d = SS1D(dim, d_state=d_state)

    def forward(self, x, h_p, w_p):
        return x + self.ss1d(self.norm(x), h_p, w_p)

class SafeMapUpsampler(nn.Module):

    def __init__(self, channels=1, max_residual=0.03, clamp_min=None, clamp_max=None):
        super().__init__()
        self.max_residual = max_residual
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.dw1 = nn.Conv2d(channels, channels, 3, 1, 0, groups=channels)
        self.act = nn.GELU()
        self.dw2 = nn.Conv2d(channels, channels, 3, 1, 0, groups=channels)
        self._init_weights()

    @staticmethod
    def _pad3(x):
        mode = 'reflect' if x.shape[-2] > 1 and x.shape[-1] > 1 else 'replicate'
        return F.pad(x, (1, 1, 1, 1), mode=mode)

    def _init_weights(self):
        nn.init.kaiming_normal_(self.dw1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.dw1.bias)
        nn.init.zeros_(self.dw2.weight)
        nn.init.zeros_(self.dw2.bias)

    def forward(self, x, size):
        base = safe_interpolate(x, size=size, mode='bilinear')
        y = self.dw1(self._pad3(base.float()))
        y = self.act(y)
        y = self.dw2(self._pad3(y))
        residual = self.max_residual * torch.tanh(y)
        out = base.float() * (1.0 + residual)
        if self.clamp_min is not None or self.clamp_max is not None:
            out = torch.clamp(out, self.clamp_min, self.clamp_max)
        return out.to(base.dtype)

class ConstrainedGridSmoother(nn.Module):

    def __init__(self, channels=3, init_alpha=0.12, max_alpha=0.25):
        super().__init__()
        self.channels = int(channels)
        self.max_alpha = float(max_alpha)

        self.raw_kernel = nn.Parameter(torch.zeros(self.channels, 1, 3, 3))

        init_ratio = float(init_alpha) / max(self.max_alpha, 1e-8)
        init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(init_ratio, dtype=torch.float32)))

    @staticmethod
    def _pad3(x):
        mode = 'reflect' if x.shape[-2] > 1 and x.shape[-1] > 1 else 'replicate'
        return F.pad(x, (1, 1, 1, 1), mode=mode)

    def forward(self, x):
        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"ConstrainedGridSmoother expects {self.channels} channels, got {C}.")

        x_float = x.float()
        x_pad = self._pad3(x_float)

        weight = F.softmax(self.raw_kernel.view(self.channels, -1), dim=1)
        weight = weight.view(self.channels, 1, 3, 3).to(dtype=x_float.dtype, device=x_float.device)

        smooth = F.conv2d(x_pad, weight, groups=self.channels)
        alpha = self.max_alpha * torch.sigmoid(self.alpha_logit)
        alpha = alpha.to(dtype=x_float.dtype, device=x_float.device)
        out = (1.0 - alpha) * x_float + alpha * smooth
        return out.to(dtype=x.dtype)

class HaMambaModule(nn.Module):

    def __init__(self, embed_dim=64, layers=1, out_channels=3):
        super().__init__()
        self.blocks = nn.ModuleList([VSSBlock1D(embed_dim) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_channels),
            nn.Sigmoid(),
        )
        nn.init.xavier_uniform_(self.head[2].weight, gain=0.1)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, tokens, input_tokens_3ch, h_p, w_p, illu_min=0.05):
        x = tokens
        for block in self.blocks:
            x = block(x, h_p, w_p)
        fea_tokens = self.head(x)
        illu_tokens = torch.clamp(input_tokens_3ch + fea_tokens, illu_min, 1.0)
        return fea_tokens, illu_tokens

class CalibrateMambaModule(nn.Module):

    def __init__(self, embed_dim=64, layers=1, in_channels=3, out_channels=3):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([VSSBlock1D(embed_dim) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_channels),
        )
        self.scale = nn.Parameter(torch.tensor(0.50))
        nn.init.xavier_uniform_(self.head[2].weight, gain=0.1)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, ref_tokens, h_p, w_p):
        x = self.norm(self.in_proj(ref_tokens))
        for block in self.blocks:
            x = block(x, h_p, w_p)
        residual = self.scale * torch.tanh(self.head(x))
        att_tokens = ref_tokens + residual
        return att_tokens

class HbMambaModule(nn.Module):

    def __init__(self, embed_dim=64, layers=1, in_channels=3, out_channels=3):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([VSSBlock1D(embed_dim) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_channels),
        )
        self.scale = nn.Parameter(torch.tensor(0.10))
        nn.init.xavier_uniform_(self.head[2].weight, gain=0.1)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, att_tokens, h_p, w_p):
        x = self.norm(self.in_proj(att_tokens))
        for block in self.blocks:
            x = block(x, h_p, w_p)
        att_1_tokens = self.scale * torch.tanh(self.head(x))
        return att_1_tokens

class FinalRefDenoiser(nn.Module):

    def __init__(self, hidden=16, max_residual=0.04):
        super().__init__()
        self.max_residual = float(max_residual)
        self.conv1 = nn.Conv2d(7, hidden, kernel_size=3, stride=1, padding=0)
        self.act1 = nn.GELU()
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=0, groups=hidden)
        self.act2 = nn.GELU()
        self.pw = nn.Conv2d(hidden, 3, kernel_size=1, stride=1, padding=0)
        self._init_weights()

    @staticmethod
    def _pad3(x):
        mode = 'reflect' if x.shape[-2] > 1 and x.shape[-1] > 1 else 'replicate'
        return F.pad(x, (1, 1, 1, 1), mode=mode)

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.dw.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.dw.bias)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, input_img, illu, ref):
        input_img = torch.clamp(input_img.float(), 0.0, 1.0)
        illu = torch.clamp(illu.float(), 0.0, 1.0)
        ref = torch.clamp(ref.float(), 0.0, 1.0)
        illu_gray = illu.mean(dim=1, keepdim=True)
        x = torch.cat([ref, input_img, illu_gray], dim=1)
        y = self.conv1(self._pad3(x))
        y = self.act1(y)
        y = self.dw(self._pad3(y))
        y = self.act2(y)
        residual = self.max_residual * torch.tanh(self.pw(y))
        out = torch.clamp(ref + residual, 0.0, 1.0)
        return out.to(dtype=ref.dtype), residual.to(dtype=ref.dtype)

class Network(nn.Module):

    def __init__(
        self,
        stage=3,
        embed_dim=64,
        patch_size=4,
        illu_min=0.05,
        retinex_eps=1e-3,
        ha_layers=1,
        calibrate_layers=1,
        hb_layers=1,
        stage_update_max=0.20,
        stage_positive_scale=0.70,
        stage_negative_scale=0.90,
        pos_init_scale=0.05,
        pos_max_scale=0.20,
        grid_init_alpha=0.12,
        grid_max_alpha=0.25,
        final_denoise_hidden=16,
        final_denoise_max_residual=0.04,
        **unused_kwargs,
    ):
        super().__init__()
        self.stage = stage
        self.patch_size = patch_size
        self.illu_min = illu_min
        self.retinex_eps = retinex_eps
        self.stage_update_max = stage_update_max
        self.stage_positive_scale = stage_positive_scale
        self.stage_negative_scale = stage_negative_scale

        self.patch_embed = PatchEmbed(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=patch_size,
            pos_init_scale=pos_init_scale,
            pos_max_scale=pos_max_scale,
        )

        self.patchify_conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=patch_size,
            stride=patch_size,
            groups=3,
            bias=False,
        )
        with torch.no_grad():
            self.patchify_conv.weight.data.fill_(1.0 / (patch_size * patch_size))
        self.patchify_conv.weight.requires_grad = False

        self.ha = HaMambaModule(embed_dim=embed_dim, layers=ha_layers, out_channels=3)
        self.calibrate = CalibrateMambaModule(embed_dim=embed_dim, layers=calibrate_layers, in_channels=3, out_channels=3)
        self.hb = HbMambaModule(embed_dim=embed_dim, layers=hb_layers, in_channels=3, out_channels=3)

        self.fea_upsampler = SafeMapUpsampler(channels=3, max_residual=0.01, clamp_min=0.0, clamp_max=1.0)
        self.illu_upsampler = SafeMapUpsampler(channels=3, max_residual=0.02, clamp_min=illu_min, clamp_max=1.0)
        self.att_upsampler = SafeMapUpsampler(channels=3, max_residual=0.01, clamp_min=None, clamp_max=None)

        self.grid_smoother = ConstrainedGridSmoother(
            channels=3,
            init_alpha=grid_init_alpha,
            max_alpha=grid_max_alpha,
        )

        self.final_ref_denoiser = FinalRefDenoiser(
            hidden=final_denoise_hidden,
            max_residual=final_denoise_max_residual,
        )


    def _tokens_to_grid(self, tokens, B, channels, h_p, w_p, clamp_min=None, clamp_max=None):
        grid = tokens.transpose(1, 2).contiguous().view(B, channels, h_p, w_p)

        if channels != 3:
            raise ValueError(f"grid_smoother expects 3-channel token maps, got channels={channels}.")
        grid = self.grid_smoother(grid).to(tokens.dtype)

        if clamp_min is not None or clamp_max is not None:
            grid = torch.clamp(grid, clamp_min, clamp_max)
        return grid

    def _tokens_to_high(self, tokens, B, channels, h_p, w_p, H, W, upsampler, clamp_min=None, clamp_max=None):
        grid = self._tokens_to_grid(tokens, B, channels, h_p, w_p, clamp_min, clamp_max)
        high = upsampler(grid, size=(H, W))
        if clamp_min is not None or clamp_max is not None:
            high = torch.clamp(high, clamp_min, clamp_max)
        return grid, high

    def _bounded_stage_update(self, raw_update_tokens):
        bounded = self.stage_update_max * torch.tanh(raw_update_tokens)
        pos = F.relu(bounded) * self.stage_positive_scale
        neg = -F.relu(-bounded) * self.stage_negative_scale
        return pos + neg

    def _retinex_reflectance(self, input_img, illu):
        illu = torch.clamp(illu.float(), self.illu_min, 1.0)
        return torch.clamp(input_img.float() / (illu + self.retinex_eps), 0.0, 1.0)

    def forward_final_illumination(self, input_img):
        B, C, H, W = input_img.shape

        tokens, h_p, w_p = self.patch_embed(input_img)
        input_grid = self.patchify_conv(input_img)
        input_tokens_3ch = input_grid.flatten(2).transpose(1, 2)

        _, illu_tokens = self.ha(
            tokens,
            input_tokens_3ch,
            h_p,
            w_p,
            self.illu_min,
        )
        ref_tokens = torch.clamp(
            input_tokens_3ch / (illu_tokens + self.retinex_eps),
            0.0,
            1.0,
        )

        for _ in range(1, self.stage):
            att_tokens = self.calibrate(ref_tokens, h_p, w_p)
            att_1_tokens = self.hb(att_tokens, h_p, w_p)

            raw_update_tokens = att_tokens + att_1_tokens
            update_tokens = self._bounded_stage_update(raw_update_tokens)
            illu_tokens = torch.clamp(
                illu_tokens + update_tokens,
                self.illu_min,
                1.0,
            )
            ref_tokens = torch.clamp(
                input_tokens_3ch / (illu_tokens + self.retinex_eps),
                0.0,
                1.0,
            )

        _, illu_high = self._tokens_to_high(
            illu_tokens,
            B,
            3,
            h_p,
            w_p,
            H,
            W,
            self.illu_upsampler,
            self.illu_min,
            1.0,
        )
        return illu_high

    def forward(self, input_img):
        B, C, H, W = input_img.shape

        tokens, h_p, w_p = self.patch_embed(input_img)
        input_grid = self.patchify_conv(input_img)
        input_tokens_3ch = input_grid.flatten(2).transpose(1, 2)

        i_list = []
        r_list = []

        _, illu_tokens = self.ha(tokens, input_tokens_3ch, h_p, w_p, self.illu_min)
        _, illu_high = self._tokens_to_high(
            illu_tokens, B, 3, h_p, w_p, H, W, self.illu_upsampler, self.illu_min, 1.0
        )
        ref_high = self._retinex_reflectance(input_img, illu_high)
        ref_tokens = torch.clamp(input_tokens_3ch / (illu_tokens + self.retinex_eps), 0.0, 1.0)

        i_list.append(illu_high)
        r_list.append(ref_high)

        for _ in range(1, self.stage):
            att_tokens = self.calibrate(ref_tokens, h_p, w_p)
            att_1_tokens = self.hb(att_tokens, h_p, w_p)
            update_tokens = self._bounded_stage_update(att_tokens + att_1_tokens)
            illu_tokens = torch.clamp(illu_tokens + update_tokens, self.illu_min, 1.0)
            _, illu_high = self._tokens_to_high(
                illu_tokens, B, 3, h_p, w_p, H, W, self.illu_upsampler, self.illu_min, 1.0
            )
            ref_high = self._retinex_reflectance(input_img, illu_high)
            ref_tokens = torch.clamp(input_tokens_3ch / (illu_tokens + self.retinex_eps), 0.0, 1.0)
            i_list.append(illu_high)
            r_list.append(ref_high)

        if len(r_list) > 0:
            r_list[-1], _ = self.final_ref_denoiser(input_img, i_list[-1], r_list[-1])

        return i_list, r_list, [input_img], [], {}, []

    @torch.no_grad()
    def inference(self, input_img):
        i_list, r_list, _, _, _, _ = self.forward(input_img)
        return r_list[-1]

    def _enhance(self, input_img, illu, return_gain=False):
        ref = self._retinex_reflectance(input_img, illu)
        if return_gain:
            gain = torch.clamp(1.0 / (torch.clamp(illu.float(), self.illu_min, 1.0) + self.retinex_eps), 1.0, 1.0 / self.illu_min)
            return ref, gain
        return ref

    @staticmethod
    def enhance_from_illu(input_img, illu, eps=1e-3, illu_min=0.05, return_gain=False):
        illu = torch.clamp(illu.float(), illu_min, 1.0)
        ref = torch.clamp(input_img.float() / (illu + eps), 0.0, 1.0)
        if return_gain:
            gain = torch.clamp(1.0 / (illu + eps), 1.0, 1.0 / illu_min)
            return ref, gain
        return ref
