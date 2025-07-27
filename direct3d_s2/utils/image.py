import numpy as np
from PIL import Image
import torch
import random
from torchvision import transforms
import torchvision.transforms.functional as TF


def apply_joint_transforms(rgb, mask, img_size, img_aug=True, test=True):
    if test:
        extra_pad = 16
    else:
        extra_pad = random.randint(0, 32)
    W_img, H_img = rgb.size[:2]
    max_HW = max(H_img, W_img)
    top_pad = (max_HW - H_img) // 2
    bottom_pad = max_HW - H_img - top_pad
    left_pad = (max_HW - W_img) // 2
    right_pad = max_HW - W_img - left_pad

    # 1. padding
    rgb = TF.pad(rgb, (left_pad, top_pad, right_pad, bottom_pad), fill=255)
    mask = TF.pad(mask, (left_pad, top_pad, right_pad, bottom_pad), fill=0) 

    if img_aug and (not test):
        # 2. random rotate
        if random.random() < 0.1:
            angle = random.uniform(-10, 10)
            rgb = TF.rotate(rgb, angle, fill=255)
            mask = TF.rotate(mask, angle, fill=0)

        # 3. random crop
        if random.random() < 0.1:
            crop_ratio = random.uniform(0.9, 1.0)
            crop_size = int(max_HW * crop_ratio)
            i, j, h, w = transforms.RandomCrop.get_params(rgb, (crop_size, crop_size))
            rgb = TF.crop(rgb, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

    # 4. resize
    target_size = (img_size, img_size)
    rgb = TF.resize(rgb, target_size, interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, target_size, interpolation=TF.InterpolationMode.NEAREST)

    # 5. extra padding
    rgb = TF.pad(rgb, extra_pad, fill=255)
    mask = TF.pad(mask, extra_pad, fill=0)
    rgb = TF.resize(rgb, target_size, interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, target_size, interpolation=TF.InterpolationMode.NEAREST)

    # to tensor
    rgb_tensor = TF.to_tensor(rgb)
    mask_tensor = TF.to_tensor(mask)
    
    return rgb_tensor, mask_tensor

def crop_recenter(image_no_bg, thereshold=100):
    image_no_bg_np = np.array(image_no_bg)
    mask = (image_no_bg_np[..., -1]).astype(np.uint8)
    mask_bin = mask > thereshold
    
    H, W = image_no_bg_np.shape[:2]
    
    valid_pixels = mask_bin.astype(np.float32).nonzero() # [N, 2]
    if np.sum(mask_bin) < (H*W) * 0.001:
        min_h =0
        max_h = H - 1
        min_w = 0
        max_w = W -1
    else:
        min_h, max_h = valid_pixels[0].min(), valid_pixels[0].max()
        min_w, max_w = valid_pixels[1].min(), valid_pixels[1].max()
        
    if min_h < 0:
        min_h = 0
    if min_w < 0:
        min_w = 0
    if max_h > H:
        max_h = H 
    if max_w > W:
        max_w = W

    image_no_bg_np = image_no_bg_np[min_h:max_h+1, min_w:max_w+1]
    return image_no_bg_np

def preprocess_image(img):

    if isinstance(img, str):
        img = Image.open(img)
        img = np.array(img)
    elif isinstance(img, Image.Image):
        img = np.array(img)

    if img.shape[-1] == 3:
        mask = np.ones_like(img[..., 0:1])
        img = np.concatenate([img, mask], axis=-1)

    img = crop_recenter(img, thereshold=0) / 255.

    mask = img[..., 3]
    img = img[..., :3] * img[..., 3:] + (1 - img[..., 3:])
    img = Image.fromarray((img * 255).astype(np.uint8))
    mask = Image.fromarray((mask * 255).astype(np.uint8))

    img, mask = apply_joint_transforms(img, mask, img_size=518, 
            img_aug=False, test=True)
    img = torch.cat([img, mask], dim=0)
    return img
    