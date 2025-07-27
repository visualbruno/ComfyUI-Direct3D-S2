import numpy as np  
import torch
from torchvision import transforms


class BiRefNet(object):
    def __init__(self, device):
        from transformers import AutoModelForImageSegmentation
        self.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet',
            trust_remote_code=True,
        ).to(device)
        self.birefnet_model.eval()
        self.device = device

    def run(self, image, use_alpha=False):
        if use_alpha:
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            return np.array(image)
        image = image.convert('RGB')
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_images = transform_image(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.birefnet_model(input_images)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        mask = np.array(mask)
        image = np.concatenate([np.array(image), mask[..., None]], axis=-1)
        return image