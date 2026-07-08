import os
import random
import shutil

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir,
        task='train',
        final_size=256,
        crop_sizes=(256, 288),
        photometric_aug=False,
        hflip_aug=True,
        vflip_aug=False,
        rotate_aug=False,
    ):
        self.low_img_dir = img_dir
        self.task = task
        self.final_size = final_size
        self.crop_sizes = list(crop_sizes) if crop_sizes is not None else [final_size]
        if len(self.crop_sizes) == 0:
            self.crop_sizes = [final_size]
        self.photometric_aug = photometric_aug
        self.hflip_aug = hflip_aug
        self.vflip_aug = vflip_aug
        self.rotate_aug = rotate_aug
        self.train_low_data_names = []
        for root, _, names in os.walk(self.low_img_dir):
            for name in names:
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                    self.train_low_data_names.append(os.path.join(root, name))
        self.train_low_data_names.sort()
        self.transform = transforms.ToTensor()

    @staticmethod
    def load_images(file):
        return Image.open(file).convert('RGB')

    def _random_crop_or_resize(self, img):
        crop_size = random.choice(self.crop_sizes)
        final_size = self.final_size
        w, h = img.size
        if w > crop_size and h > crop_size:
            i = random.randint(0, h - crop_size)
            j = random.randint(0, w - crop_size)
            img = TF.crop(img, i, j, crop_size, crop_size)
        elif w != crop_size or h != crop_size:
            img = TF.resize(img, (crop_size, crop_size))
        if crop_size != final_size:
            img = TF.resize(img, (final_size, final_size))
        return img

    def _geometry_aug(self, img):
        if self.hflip_aug and random.random() > 0.5:
            img = TF.hflip(img)
        if self.vflip_aug and random.random() > 0.5:
            img = TF.vflip(img)
        if self.rotate_aug and random.random() > 0.5:
            img = TF.rotate(img, random.choice([90, 180, 270]))
        return img

    @staticmethod
    def _photometric_aug(img):
        if random.random() > 0.7:
            img = TF.adjust_contrast(img, random.uniform(0.95, 1.05))
        if random.random() > 0.7:
            img = TF.adjust_brightness(img, random.uniform(0.95, 1.05))
        if random.random() > 0.8:
            img = TF.adjust_saturation(img, random.uniform(0.95, 1.05))
        if random.random() > 0.85:
            img = TF.adjust_gamma(img, random.uniform(0.95, 1.10))
        return img

    def __getitem__(self, index):
        img_name = self.train_low_data_names[index]
        low_im = self.load_images(img_name)
        if self.task == 'train':
            low_im = self._random_crop_or_resize(low_im)
            low_im = self._geometry_aug(low_im)
            if self.photometric_aug:
                low_im = self._photometric_aug(low_im)
        low_tensor = self.transform(low_im)
        return low_tensor, img_name

    def __len__(self):
        return len(self.train_low_data_names)


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.cnt += n
        self.avg = self.sum / self.cnt if self.cnt != 0 else 0.0


def count_parameters_in_MB(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6


def create_exp_dir(path, scripts_to_save=None):
    os.makedirs(path, exist_ok=True)
    if scripts_to_save is not None:
        script_path = os.path.join(path, 'scripts')
        os.makedirs(script_path, exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(script_path, os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(state, is_best, save):
    os.makedirs(save, exist_ok=True)
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path, strict=True, map_location='cpu'):
    state_dict = torch.load(model_path, map_location=map_location)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=strict)


def normalize_image_numpy(image_numpy, eps=1e-8):
    x_min = image_numpy.min()
    x_max = image_numpy.max()
    return (image_numpy - x_min) / (x_max - x_min + eps)


def tensor_to_image_numpy(tensor, normalize=False):
    if tensor is None:
        raise ValueError('Input tensor is None.')
    if not torch.is_tensor(tensor):
        raise TypeError('Input must be a torch.Tensor.')
    tensor = tensor.detach().float().cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]
    elif tensor.dim() != 3:
        raise ValueError(f'Unsupported tensor dim: {tensor.dim()}, expected 3 or 4.')
    image_numpy = tensor.numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = image_numpy[0]
        image_numpy = normalize_image_numpy(image_numpy) if normalize else np.clip(image_numpy, 0.0, 1.0)
    elif image_numpy.shape[0] == 3:
        image_numpy = normalize_image_numpy(image_numpy) if normalize else np.clip(image_numpy, 0.0, 1.0)
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:
        raise ValueError(f'Unsupported channel count: {image_numpy.shape[0]}, expected 1 or 3.')
    return image_numpy


def _save_numpy_image(image_numpy, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image_uint8 = (np.clip(image_numpy, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image_uint8).save(path, 'png')


def save_images(tensor, path, normalize=False):
    image_numpy = tensor_to_image_numpy(tensor, normalize=normalize)
    _save_numpy_image(image_numpy, path)


def save_feature_map(tensor, path):
    save_images(tensor, path, normalize=True)


def save_rgb_image(tensor, path):
    save_images(tensor, path, normalize=False)


def get_last_checkpoint(save_dir, suffix='.pth'):
    if not os.path.exists(save_dir):
        return None
    files = sorted([f for f in os.listdir(save_dir) if f.endswith(suffix)])
    if len(files) == 0:
        return None
    return os.path.join(save_dir, files[-1])


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def model_info(model):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    return {
        'total_params': n_p,
        'trainable_params': n_g,
        'total_params_MB': n_p / 1e6,
        'trainable_params_MB': n_g / 1e6,
    }
