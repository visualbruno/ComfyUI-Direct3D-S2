import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class DinoEncoder(nn.Module):

    def __init__(
            self,
            model="facebookresearch/dinov2",
            version="dinov2_vitl14_reg",
            size=518,
    ):
        super().__init__()

        dino_model = torch.hub.load(model, version, pretrained=True)
        dino_model = dino_model.eval()
        self.encoder = dino_model
        self.transform = transforms.Compose(
            [
                transforms.Resize(size, transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.CenterCrop(size), 
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def forward(self, image, image_mask=None):

        z = self.encoder(self.transform(image), is_training=True)['x_prenorm']
        z = F.layer_norm(z, z.shape[-1:])

        if image_mask is not None:
            image_mask_patch = F.max_pool2d(image_mask, kernel_size=14, stride=14).squeeze(1) > 0
            return z, image_mask_patch

        return z
