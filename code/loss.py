
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None

class SmoothLoss(nn.Module):

    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 10

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        mat = input_im.new_tensor(
            [[0.257, -0.148, 0.439],
             [0.564, -0.291, -0.368],
             [0.098, 0.439, -0.071]]
        ).float()
        bias = input_im.new_tensor(
            [16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]
        ).float()
        temp = im_flat.mm(mat) + bias
        out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])
        return out

    def forward(self, input, output):
        self.output = output
        self.input = self.rgb2yCbCr(input)
        sigma_color = -1.0 / (2 * self.sigma * self.sigma)

        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        p = 1.0

        pixel_grad1 = w1 * torch.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p, dim=1, keepdim=True)
        pixel_grad2 = w2 * torch.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p, dim=1, keepdim=True)
        pixel_grad3 = w3 * torch.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p, dim=1, keepdim=True)
        pixel_grad4 = w4 * torch.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p, dim=1, keepdim=True)
        pixel_grad5 = w5 * torch.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p, dim=1, keepdim=True)
        pixel_grad6 = w6 * torch.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p, dim=1, keepdim=True)
        pixel_grad7 = w7 * torch.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p, dim=1, keepdim=True)
        pixel_grad8 = w8 * torch.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p, dim=1, keepdim=True)
        pixel_grad9 = w9 * torch.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p, dim=1, keepdim=True)
        pixel_grad10 = w10 * torch.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p, dim=1, keepdim=True)
        pixel_grad11 = w11 * torch.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p, dim=1, keepdim=True)
        pixel_grad12 = w12 * torch.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p, dim=1, keepdim=True)
        pixel_grad13 = w13 * torch.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p, dim=1, keepdim=True)
        pixel_grad14 = w14 * torch.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p, dim=1, keepdim=True)
        pixel_grad15 = w15 * torch.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p, dim=1, keepdim=True)
        pixel_grad16 = w16 * torch.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p, dim=1, keepdim=True)
        pixel_grad17 = w17 * torch.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p, dim=1, keepdim=True)
        pixel_grad18 = w18 * torch.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p, dim=1, keepdim=True)
        pixel_grad19 = w19 * torch.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p, dim=1, keepdim=True)
        pixel_grad20 = w20 * torch.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p, dim=1, keepdim=True)
        pixel_grad21 = w21 * torch.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p, dim=1, keepdim=True)
        pixel_grad22 = w22 * torch.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p, dim=1, keepdim=True)
        pixel_grad23 = w23 * torch.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p, dim=1, keepdim=True)
        pixel_grad24 = w24 * torch.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p, dim=1, keepdim=True)

        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)
        total_term = ReguTerm1
        return total_term

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        return torch.pow(Drg + Drb + Dgb + 1e-12, 0.5)

class TargetExtractionLoss(nn.Module):

    def __init__(
        self,
        median_kernel: int = 3,
        gaussian_sigma: float = 0.6,
        bright_otsu_scale: float = 0.8,
        bright_soft_high_percentile: float = 99.5,
        bright_soft_blur_sigma: float = 0.8,
        texture_otsu_scale: float = 0.70,
        edge_low_percentile: float = 80.0,
        edge_high_percentile: float = 99.0,
        edge_soft_blur_sigma: float = 0.6,
        seed_dilate_radius: int = 2,
        seed_close_radius: int = 3,
        fill_dilate_radius: int = 1,
        min_area: int = 180,
        feature_region_blur_sigma: float = 1.2,
        feature_region_base: float = 0.30,
        feature_edge_gain: float = 0.70,
        strength_blur_sigma: float = 0.8,
        target_illu_min: float = 0.10,
        target_illu_max: float = 1.00,
        dark_feature_weight: float = 1.45,
        bright_feature_weight: float = 0.15,
        background_pull: float = 0.65,
        dark_region_weight: float = 2.40,
        bright_region_weight: float = 0.25,
        background_region_weight: float = 0.45,
        loss_type: str = 'charbonnier',
        charbonnier_eps: float = 1e-3,
    ):
        super().__init__()
        if cv2 is None:
            raise ImportError(
                "TargetExtractionLoss requires OpenCV. "
                "Install opencv-python or opencv-python-headless."
            )

        self.median_kernel = int(median_kernel)
        if self.median_kernel % 2 == 0:
            self.median_kernel += 1
        self.gaussian_sigma = float(gaussian_sigma)

        self.bright_otsu_scale = float(bright_otsu_scale)
        self.bright_soft_high_percentile = float(bright_soft_high_percentile)
        self.bright_soft_blur_sigma = float(bright_soft_blur_sigma)

        self.texture_otsu_scale = float(texture_otsu_scale)
        self.edge_low_percentile = float(edge_low_percentile)
        self.edge_high_percentile = float(edge_high_percentile)
        self.edge_soft_blur_sigma = float(edge_soft_blur_sigma)

        self.seed_dilate_radius = int(seed_dilate_radius)
        self.seed_close_radius = int(seed_close_radius)
        self.fill_dilate_radius = int(fill_dilate_radius)
        self.min_area = int(min_area)

        self.feature_region_blur_sigma = float(feature_region_blur_sigma)
        self.feature_region_base = float(feature_region_base)
        self.feature_edge_gain = float(feature_edge_gain)

        self.strength_blur_sigma = float(strength_blur_sigma)

        self.target_illu_min = float(target_illu_min)
        self.target_illu_max = float(target_illu_max)

        self.dark_feature_weight = float(dark_feature_weight)
        self.bright_feature_weight = float(bright_feature_weight)
        self.background_pull = float(background_pull)

        self.dark_region_weight = float(dark_region_weight)
        self.bright_region_weight = float(bright_region_weight)
        self.background_region_weight = float(background_region_weight)

        self.loss_type = str(loss_type).lower()
        self.charbonnier_eps = float(charbonnier_eps)

    @staticmethod
    def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    @staticmethod
    def _normalize01_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        x = x.astype(np.float32)
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        return ((x - x_min) / max(x_max - x_min, eps)).astype(np.float32)

    @staticmethod
    def _safe_percentile(x: np.ndarray, q: float, fallback: float = 0.0) -> float:
        x = np.asarray(x, dtype=np.float32)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float(fallback)
        return float(np.percentile(x, q))

    @staticmethod
    def _soft_map_from_threshold(x: np.ndarray, low: float, high: float, eps: float = 1e-6) -> np.ndarray:
        x = x.astype(np.float32)
        if high <= low + eps:
            return (x > low).astype(np.float32)
        y = (x - low) / (high - low + eps)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _disk(radius: int) -> np.ndarray:
        radius = int(radius)
        if radius <= 0:
            return np.ones((1, 1), dtype=np.uint8)
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = (x * x + y * y) <= (radius * radius + 1e-6)
        return mask.astype(np.uint8)

    @staticmethod
    def _otsu_threshold01(x01: np.ndarray) -> float:
        x_u8 = np.clip(np.round(x01 * 255.0), 0, 255).astype(np.uint8)
        level, _ = cv2.threshold(x_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(level) / 255.0

    @staticmethod
    def _gaussian_blur01(x01: np.ndarray, sigma: float) -> np.ndarray:
        x01 = np.clip(x01.astype(np.float32), 0.0, 1.0)
        if sigma <= 0:
            return x01
        y = cv2.GaussianBlur(
            x01,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        return np.clip(y.astype(np.float32), 0.0, 1.0)

    def _remove_small_components(self, mask_bool: np.ndarray) -> np.ndarray:
        mask_u8 = (mask_bool > 0).astype(np.uint8)
        if self.min_area <= 1:
            return mask_u8
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        clean = np.zeros_like(mask_u8)
        for lab in range(1, num_labels):
            if stats[lab, cv2.CC_STAT_AREA] >= self.min_area:
                clean[labels == lab] = 1
        return clean.astype(np.uint8)

    @staticmethod
    def _fill_holes(mask_bool: np.ndarray) -> np.ndarray:
        mask = (mask_bool > 0).astype(np.uint8)
        inv = (1 - mask).astype(np.uint8)
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        if num_labels <= 1:
            return mask

        border_labels = set()
        border_labels.update(np.unique(labels[0, :]).tolist())
        border_labels.update(np.unique(labels[-1, :]).tolist())
        border_labels.update(np.unique(labels[:, 0]).tolist())
        border_labels.update(np.unique(labels[:, -1]).tolist())

        holes = np.zeros_like(mask, dtype=np.uint8)
        for lab in range(1, num_labels):
            if lab not in border_labels:
                holes[labels == lab] = 1
        return np.logical_or(mask > 0, holes > 0).astype(np.uint8)

    def _build_bright_branch(self, base_gray: np.ndarray) -> dict:

        level_bright = self._otsu_threshold01(base_gray)
        bright_threshold = float(level_bright * self.bright_otsu_scale)
        bright_mask_hard = (base_gray > bright_threshold).astype(np.float32)

        valid_values = base_gray[base_gray > bright_threshold]
        if valid_values.size > 0:
            bright_high = self._safe_percentile(
                valid_values,
                self.bright_soft_high_percentile,
                fallback=max(bright_threshold + 1e-3, float(np.max(base_gray))),
            )
        else:
            bright_high = self._safe_percentile(
                base_gray,
                self.bright_soft_high_percentile,
                fallback=bright_threshold + 1e-3,
            )

        bright_soft_raw = self._soft_map_from_threshold(base_gray, low=bright_threshold, high=bright_high)
        bright_soft_x = self._gaussian_blur01(bright_soft_raw, sigma=self.bright_soft_blur_sigma)

        return {
            "bright_mask_hard": bright_mask_hard.astype(np.float32),
            "bright_soft_raw": bright_soft_raw.astype(np.float32),
            "bright_soft_x": bright_soft_x.astype(np.float32),
        }

    def _build_feature_y_branch(self, base_gray: np.ndarray) -> dict:

        gy, gx = np.gradient(base_gray.astype(np.float32))
        grad_raw = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        grad_norm = self._normalize01_np(grad_raw)

        level_texture = self._otsu_threshold01(grad_norm)
        texture_threshold = max(float(level_texture * self.texture_otsu_scale), 1e-6)

        edge_seed_hard = (grad_norm > texture_threshold).astype(np.uint8)

        if self.seed_dilate_radius > 0:
            edge_seed_dilated = cv2.dilate(edge_seed_hard, self._disk(self.seed_dilate_radius), iterations=1)
        else:
            edge_seed_dilated = edge_seed_hard.copy()
        edge_seed_dilated = (edge_seed_dilated > 0).astype(np.uint8)

        if self.seed_close_radius > 0:
            edge_seed_closed = cv2.morphologyEx(
                edge_seed_dilated,
                cv2.MORPH_CLOSE,
                self._disk(self.seed_close_radius),
                iterations=1,
            )
        else:
            edge_seed_closed = edge_seed_dilated.copy()
        edge_seed_closed = (edge_seed_closed > 0).astype(np.uint8)

        edge_seed_filled = self._fill_holes(edge_seed_closed)

        if self.fill_dilate_radius > 0:
            feature_region_expanded = cv2.dilate(edge_seed_filled, self._disk(self.fill_dilate_radius), iterations=1)
        else:
            feature_region_expanded = edge_seed_filled.copy()
        feature_region_expanded = (feature_region_expanded > 0).astype(np.uint8)

        feature_region_clean = self._remove_small_components(feature_region_expanded)

        if self.seed_close_radius > 0:
            feature_region_hard = cv2.morphologyEx(
                feature_region_clean,
                cv2.MORPH_CLOSE,
                self._disk(self.seed_close_radius),
                iterations=1,
            )
        else:
            feature_region_hard = feature_region_clean.copy()
        feature_region_hard = (feature_region_hard > 0).astype(np.float32)

        feature_region_soft = self._gaussian_blur01(feature_region_hard, sigma=self.feature_region_blur_sigma)

        positive_grad = grad_norm[grad_norm > 0]
        if positive_grad.size > 0:
            edge_low = self._safe_percentile(
                positive_grad,
                self.edge_low_percentile,
                fallback=texture_threshold,
            )
            edge_high = self._safe_percentile(
                positive_grad,
                self.edge_high_percentile,
                fallback=max(texture_threshold + 1e-3, edge_low + 1e-3),
            )
        else:
            edge_low = texture_threshold
            edge_high = texture_threshold + 1e-3
        edge_low = max(edge_low, texture_threshold)

        edge_soft_raw = self._soft_map_from_threshold(grad_norm, low=edge_low, high=edge_high)
        edge_soft_raw = edge_soft_raw * feature_region_hard

        edge_soft = self._gaussian_blur01(edge_soft_raw, sigma=self.edge_soft_blur_sigma)
        edge_soft = np.clip(edge_soft * feature_region_soft, 0.0, 1.0).astype(np.float32)

        y_feature_raw = (
            feature_region_soft
            * (self.feature_region_base + self.feature_edge_gain * edge_soft)
        ).astype(np.float32)
        y_feature = np.clip(y_feature_raw, 0.0, 1.0).astype(np.float32)

        return {
            "grad_norm": grad_norm.astype(np.float32),
            "edge_seed_hard": edge_seed_hard.astype(np.float32),
            "edge_seed_dilated": edge_seed_dilated.astype(np.float32),
            "edge_seed_closed": edge_seed_closed.astype(np.float32),
            "edge_seed_filled": edge_seed_filled.astype(np.float32),
            "feature_region_expanded": feature_region_expanded.astype(np.float32),
            "feature_region_clean": feature_region_clean.astype(np.float32),
            "feature_region_hard": feature_region_hard.astype(np.float32),
            "feature_region_soft": feature_region_soft.astype(np.float32),
            "edge_soft_raw": edge_soft_raw.astype(np.float32),
            "edge_soft": edge_soft.astype(np.float32),
            "y_feature_raw": y_feature_raw.astype(np.float32),
            "y_feature": y_feature.astype(np.float32),
        }

    def _process_one_gray(self, gray01: np.ndarray) -> dict:
        gray01 = np.clip(gray01.astype(np.float32), 0.0, 1.0)
        gray_u8 = np.clip(np.round(gray01 * 255.0), 0, 255).astype(np.uint8)

        if self.median_kernel > 1:
            median_u8 = cv2.medianBlur(gray_u8, self.median_kernel)
        else:
            median_u8 = gray_u8

        median_gray = median_u8.astype(np.float32) / 255.0
        base_gray = self._gaussian_blur01(median_gray, sigma=self.gaussian_sigma)

        bright_maps = self._build_bright_branch(base_gray)
        feature_maps = self._build_feature_y_branch(base_gray)

        x = bright_maps["bright_soft_x"].astype(np.float32)
        y = feature_maps["y_feature"].astype(np.float32)

        feature_region_soft = feature_maps["feature_region_soft"].astype(np.float32)
        background_mask = (1.0 - feature_region_soft).astype(np.float32)

        dark_feature = ((1.0 - x) * y).astype(np.float32)
        bright_feature = (x * y).astype(np.float32)

        weighted_dark_feature = (self.dark_feature_weight * dark_feature).astype(np.float32)
        weighted_bright_feature = (self.bright_feature_weight * bright_feature).astype(np.float32)

        target_strength_raw = np.clip(
            weighted_dark_feature + weighted_bright_feature,
            0.0,
            1.0,
        ).astype(np.float32)

        target_strength_smooth = self._gaussian_blur01(target_strength_raw, sigma=self.strength_blur_sigma)
        target_strength = np.clip(target_strength_smooth * feature_region_soft, 0.0, 1.0).astype(np.float32)

        target_illu_raw = (
            self.target_illu_max
            - (self.target_illu_max - self.target_illu_min) * target_strength
        ).astype(np.float32)

        if self.background_pull > 0:
            target_illu_bg_pulled = (
                target_illu_raw
                + self.background_pull
                * background_mask
                * (self.target_illu_max - target_illu_raw)
            ).astype(np.float32)
        else:
            target_illu_bg_pulled = target_illu_raw.copy()

        target_illu = np.clip(target_illu_bg_pulled, self.target_illu_min, self.target_illu_max).astype(np.float32)
        target_illu_gray_texture_preview = np.clip(
            target_illu + (1.0 - target_illu) * gray01,
            0.0,
            1.0,
        ).astype(np.float32)

        loss_weight = (
            self.dark_region_weight * dark_feature
            + self.bright_region_weight * bright_feature
            + self.background_region_weight * background_mask
        ).astype(np.float32)

        return {
            "gray": gray01.astype(np.float32),
            "median_gray": median_gray.astype(np.float32),
            "base_gray": base_gray.astype(np.float32),
            **bright_maps,
            **feature_maps,
            "background_mask": background_mask.astype(np.float32),
            "dark_feature": dark_feature.astype(np.float32),
            "bright_feature": bright_feature.astype(np.float32),
            "weighted_dark_feature": weighted_dark_feature.astype(np.float32),
            "weighted_bright_feature": weighted_bright_feature.astype(np.float32),
            "target_strength_raw": target_strength_raw.astype(np.float32),
            "target_strength_smooth": target_strength_smooth.astype(np.float32),
            "target_strength": target_strength.astype(np.float32),
            "target_illu_raw": target_illu_raw.astype(np.float32),
            "target_illu_bg_pulled": target_illu_bg_pulled.astype(np.float32),
            "target_illu": target_illu.astype(np.float32),
            "target_illu_gray_texture_preview": target_illu_gray_texture_preview.astype(np.float32),
            "loss_weight": loss_weight.astype(np.float32),
        }

    @staticmethod
    def _stack_to_tensor(batch_maps: list, key: str, device, dtype) -> torch.Tensor:
        arr = np.stack([m[key] for m in batch_maps], axis=0)[:, None, :, :]
        return torch.from_numpy(arr).to(device=device, dtype=dtype)

    def build_guidance(self, input_img: torch.Tensor) -> dict:
        input_img = torch.clamp(input_img.detach().float(), 0.0, 1.0)
        gray = self.rgb_to_gray(input_img)
        device = input_img.device
        dtype = input_img.dtype
        gray_np = gray.detach().cpu().numpy()[:, 0, :, :]
        batch_maps = [self._process_one_gray(gray_np[b]) for b in range(gray_np.shape[0])]
        keys = list(batch_maps[0].keys())
        return {
            key: self._stack_to_tensor(batch_maps, key, device, dtype).detach()
            for key in keys
        }

    def _pixel_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        if self.loss_type in ('l1', 'mae'):
            return torch.abs(diff)
        return torch.sqrt(diff * diff + self.charbonnier_eps * self.charbonnier_eps)

    @staticmethod
    def _masked_mean(err: torch.Tensor, mask: torch.Tensor):
        denom = mask.sum()
        return (err * mask).sum() / torch.clamp(denom, min=1e-6), denom

    def _regression_loss(self, pred: torch.Tensor, target: torch.Tensor, guidance: dict) -> torch.Tensor:
        err = self._pixel_error(pred, target)
        dark_mask = guidance["dark_feature"].to(device=pred.device, dtype=pred.dtype)
        bright_mask = guidance["bright_feature"].to(device=pred.device, dtype=pred.dtype)
        bg_mask = guidance["background_mask"].to(device=pred.device, dtype=pred.dtype)

        dark_loss, dark_count = self._masked_mean(err, dark_mask)
        bright_loss, bright_count = self._masked_mean(err, bright_mask)
        bg_loss, bg_count = self._masked_mean(err, bg_mask)

        active_dark = (dark_count > 0).to(dtype=pred.dtype)
        active_bright = (bright_count > 0).to(dtype=pred.dtype)
        active_bg = (bg_count > 0).to(dtype=pred.dtype)

        wd = pred.new_tensor(self.dark_region_weight)
        wb = pred.new_tensor(self.bright_region_weight)
        wbg = pred.new_tensor(self.background_region_weight)

        total_weight = wd * active_dark + wb * active_bright + wbg * active_bg
        total = wd * active_dark * dark_loss + wb * active_bright * bright_loss + wbg * active_bg * bg_loss
        return total / torch.clamp(total_weight, min=1e-6)

    def _format_inter_processes(self, guidance: dict, device, dtype, abs_error: torch.Tensor = None) -> dict:
        ordered_keys = [
            ("target_01_gray", "gray"),
            ("target_02_median_gray", "median_gray"),
            ("target_03_base_gray", "base_gray"),
            ("target_04_bright_mask_hard", "bright_mask_hard"),
            ("target_05_bright_soft_raw", "bright_soft_raw"),
            ("target_06_bright_soft_x", "bright_soft_x"),
            ("target_07_grad_norm", "grad_norm"),
            ("target_08_edge_seed_hard", "edge_seed_hard"),
            ("target_09_edge_seed_dilated", "edge_seed_dilated"),
            ("target_10_edge_seed_closed", "edge_seed_closed"),
            ("target_11_edge_seed_filled", "edge_seed_filled"),
            ("target_12_feature_region_expanded", "feature_region_expanded"),
            ("target_13_feature_region_clean", "feature_region_clean"),
            ("target_14_feature_region_hard", "feature_region_hard"),
            ("target_15_feature_region_soft", "feature_region_soft"),
            ("target_16_edge_soft_raw", "edge_soft_raw"),
            ("target_17_edge_soft", "edge_soft"),
            ("target_18_y_feature_raw", "y_feature_raw"),
            ("target_19_y_feature_final", "y_feature"),
            ("target_20_background_mask", "background_mask"),
            ("target_21_dark_feature_1_minus_x_mul_y", "dark_feature"),
            ("target_22_bright_feature_x_mul_y", "bright_feature"),
            ("target_23_weighted_dark_feature", "weighted_dark_feature"),
            ("target_24_weighted_bright_feature", "weighted_bright_feature"),
            ("target_25_target_strength_raw", "target_strength_raw"),
            ("target_26_target_strength_smooth", "target_strength_smooth"),
            ("target_27_target_strength", "target_strength"),
            ("target_28_target_illu_raw", "target_illu_raw"),
            ("target_29_target_illu_bg_pulled", "target_illu_bg_pulled"),
            ("target_30_target_illu_final", "target_illu"),
            ("target_31_target_illu_gray_texture_preview", "target_illu_gray_texture_preview"),
        ]
        inter_processes = {
            out_key: guidance[src_key].to(device=device, dtype=dtype)
            for out_key, src_key in ordered_keys
            if src_key in guidance
        }
        if abs_error is not None:
            inter_processes["target_32_abs_error_illu_gray_to_target"] = abs_error
        return inter_processes

    def forward(self, input_img, enhanced_img, illu, guidance=None):
        del enhanced_img
        input_img = torch.clamp(input_img.float(), 0.0, 1.0)
        if guidance is None:
            guidance = self.build_guidance(input_img)

        illu_gray = torch.mean(torch.clamp(illu.float(), 0.0, 1.0), dim=1, keepdim=True)
        target_illu = guidance["target_illu"].to(device=illu.device, dtype=illu.dtype)
        loss_weight = guidance["loss_weight"].to(device=illu.device, dtype=illu.dtype)

        total_target_loss = self._regression_loss(illu_gray, target_illu, guidance)
        abs_error = torch.abs(illu_gray - target_illu) * (loss_weight > 0).to(dtype=illu_gray.dtype)
        inter_processes = self._format_inter_processes(
            guidance,
            device=illu.device,
            dtype=illu.dtype,
            abs_error=abs_error,
        )
        return total_target_loss, inter_processes

class LossFunction(nn.Module):

    def __init__(
        self,
        w_prior: float = 0.004,
        w_smooth: float = 0.002,
        w_illu_color: float = 0.004,
        w_ref_color: float = 0.38,
        w_target: float = 3.5,
        eps: float = 1e-3,
        illu_min: float = 0.05,
        target_final_scale: float = 1.00,
        target_early_scale: float = 0.40,
        target_middle_scale: float = 0.70,
        target_stage_mode: str = 'all',
        target_median_kernel: int = 3,
        target_gaussian_sigma: float = 0.6,
        target_bright_otsu_scale: float = 0.8,
        target_bright_soft_high_percentile: float = 99.5,
        target_bright_soft_blur_sigma: float = 0.8,
        target_texture_otsu_scale: float = 0.70,
        target_edge_low_percentile: float = 80.0,
        target_edge_high_percentile: float = 99.0,
        target_edge_soft_blur_sigma: float = 0.6,
        target_seed_dilate_radius: int = 2,
        target_seed_close_radius: int = 3,
        target_fill_dilate_radius: int = 1,
        target_min_area: int = 180,
        target_feature_region_blur_sigma: float = 1.2,
        target_feature_region_base: float = 0.30,
        target_feature_edge_gain: float = 0.70,
        target_strength_blur_sigma: float = 0.8,
        target_illu_min: float = 0.10,
        target_illu_max: float = 1.00,
        target_dark_feature_weight: float = 1.45,
        target_bright_feature_weight: float = 0.15,
        target_background_pull: float = 0.65,
        target_dark_region_weight: float = 2.40,
        target_bright_region_weight: float = 0.25,
        target_background_region_weight: float = 0.45,
        target_loss_type: str = 'charbonnier',
        ref_color_ratio_min: float = 0.50,
        ref_color_ratio_max: float = 1.50,
        ref_color_spread_weight: float = 0.10,
        ref_color_extreme_weight: float = 2.00,
    ):
        super().__init__()
        self.w_prior = float(w_prior)
        self.w_smooth = float(w_smooth)
        self.w_illu_color = float(w_illu_color)
        self.w_ref_color = float(w_ref_color)
        self.w_target = float(w_target)
        self.eps = float(eps)
        self.illu_min = float(illu_min)

        self.ref_color_ratio_min = float(ref_color_ratio_min)
        self.ref_color_ratio_max = float(ref_color_ratio_max)
        self.ref_color_spread_weight = float(ref_color_spread_weight)
        self.ref_color_extreme_weight = float(ref_color_extreme_weight)
        if self.ref_color_ratio_max <= self.ref_color_ratio_min:
            raise ValueError(
                f"ref_color_ratio_max must be larger than ref_color_ratio_min, "
                f"got min={self.ref_color_ratio_min}, max={self.ref_color_ratio_max}"
            )

        self.target_final_scale = float(target_final_scale)
        self.target_early_scale = float(target_early_scale)
        self.target_middle_scale = float(target_middle_scale)
        self.target_stage_mode = target_stage_mode
        self.target_global_scale = 1.0

        self.l2_loss = nn.MSELoss()
        self.smooth_loss = SmoothLoss()
        self.color_loss = L_color()
        self.target_loss = TargetExtractionLoss(
            median_kernel=target_median_kernel,
            gaussian_sigma=target_gaussian_sigma,
            bright_otsu_scale=target_bright_otsu_scale,
            bright_soft_high_percentile=target_bright_soft_high_percentile,
            bright_soft_blur_sigma=target_bright_soft_blur_sigma,
            texture_otsu_scale=target_texture_otsu_scale,
            edge_low_percentile=target_edge_low_percentile,
            edge_high_percentile=target_edge_high_percentile,
            edge_soft_blur_sigma=target_edge_soft_blur_sigma,
            seed_dilate_radius=target_seed_dilate_radius,
            seed_close_radius=target_seed_close_radius,
            fill_dilate_radius=target_fill_dilate_radius,
            min_area=target_min_area,
            feature_region_blur_sigma=target_feature_region_blur_sigma,
            feature_region_base=target_feature_region_base,
            feature_edge_gain=target_feature_edge_gain,
            strength_blur_sigma=target_strength_blur_sigma,
            target_illu_min=target_illu_min,
            target_illu_max=target_illu_max,
            dark_feature_weight=target_dark_feature_weight,
            bright_feature_weight=target_bright_feature_weight,
            background_pull=target_background_pull,
            dark_region_weight=target_dark_region_weight,
            bright_region_weight=target_bright_region_weight,
            background_region_weight=target_background_region_weight,
            loss_type=target_loss_type,
        )

    def _target_scale_for_stage(self, stage_mode: str) -> float:
        if self.target_stage_mode == 'early_only':
            return self.target_early_scale if stage_mode == 'early' else 0.0
        if self.target_stage_mode == 'final_only':
            return self.target_final_scale if stage_mode == 'final' else 0.0
        if self.target_stage_mode == 'all':
            if stage_mode == 'early':
                return self.target_early_scale
            if stage_mode == 'middle':
                return self.target_middle_scale
            if stage_mode == 'final':
                return self.target_final_scale
            return 0.0
        if stage_mode == 'early':
            return self.target_early_scale
        if stage_mode == 'final':
            return self.target_final_scale
        return 0.0

    def _safe_guidance(self, input_img: torch.Tensor) -> dict:
        with torch.no_grad():
            return self.target_loss.build_guidance(input_img)

    def _ratio_stability_loss(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        image_gray = torch.mean(image, dim=1, keepdim=True)
        reference_gray = torch.mean(reference, dim=1, keepdim=True)

        image_ratio_raw = image / (image_gray + self.eps)
        reference_ratio_raw = (reference / (reference_gray + self.eps)).detach()

        ratio_min = self.ref_color_ratio_min
        ratio_max = self.ref_color_ratio_max

        image_ratio_core = torch.clamp(image_ratio_raw, ratio_min, ratio_max)
        reference_ratio_core = torch.clamp(reference_ratio_raw, ratio_min, ratio_max)

        ratio_align = torch.mean(torch.abs(image_ratio_core - reference_ratio_core))

        low_over = F.relu(ratio_min - image_ratio_raw)
        high_over = F.relu(image_ratio_raw - ratio_max)
        extreme_penalty = torch.mean(low_over.pow(2) + high_over.pow(2))

        ratio_spread = torch.mean(
            torch.abs(image_ratio_core - image_ratio_core.mean(dim=1, keepdim=True))
        )

        return (
            ratio_align
            + self.ref_color_spread_weight * ratio_spread
            + self.ref_color_extreme_weight * extreme_penalty
        )

    def forward(
        self,
        input_img: torch.Tensor,
        illu: torch.Tensor,
        enhanced: torch.Tensor = None,
        stage_mode: str = 'final',
        guidance: dict = None,
    ):
        input_img = torch.clamp(input_img.float(), 0.0, 1.0)
        illu = torch.clamp(illu.float(), self.illu_min, 1.0)

        if enhanced is None:
            raise ValueError("LossFunction requires `enhanced` from the model. In train.py this should be ref_s.")
        enhanced = torch.clamp(enhanced.float(), 0.0, 1.0)

        if guidance is None:
            guidance = self._safe_guidance(input_img)

        prior = self.l2_loss(enhanced, input_img)
        smooth = self.smooth_loss(input_img, enhanced)
        illu_color = torch.mean(self.color_loss(illu))
        ref_color = self._ratio_stability_loss(enhanced, input_img)

        if stage_mode == 'early':
            ref_color_w = 0.25 * self.w_ref_color
        elif stage_mode == 'middle':
            ref_color_w = 0.70 * self.w_ref_color
        else:
            ref_color_w = self.w_ref_color

        prior_weighted = self.w_prior * prior
        smooth_weighted = self.w_smooth * smooth
        illu_color_weighted = self.w_illu_color * illu_color
        ref_color_weighted = ref_color_w * ref_color

        total = prior_weighted + smooth_weighted + illu_color_weighted + ref_color_weighted

        sub_losses = {
            "prior": prior.detach(),
            "smooth": smooth.detach(),
            "illu_color": illu_color.detach(),
            "ref_color": ref_color.detach(),
            "prior_w": prior_weighted.detach(),
            "smooth_w": smooth_weighted.detach(),
            "illu_color_w": illu_color_weighted.detach(),
            "ref_color_w": ref_color_weighted.detach(),
        }

        loss_inter_maps = {}
        target_scale = self._target_scale_for_stage(stage_mode) * self.target_global_scale
        if self.w_target > 0 and target_scale > 0:
            target_loss_val, loss_inter_maps = self.target_loss(input_img, enhanced, illu, guidance=guidance)
            target_weighted = (self.w_target * target_scale) * target_loss_val
            total = total + target_weighted
            sub_losses["target"] = target_loss_val.detach()
            sub_losses["target_w"] = target_weighted.detach()
        else:
            zero = torch.zeros((), device=input_img.device, dtype=input_img.dtype)
            sub_losses["target"] = zero
            sub_losses["target_w"] = zero

        return total, sub_losses, loss_inter_maps
