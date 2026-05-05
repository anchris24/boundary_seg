import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_BN_Weights, vgg16_bn


def _conv_block(in_channels: int, out_channels: int, num_convs: int) -> nn.Sequential:
    layers = []
    current_in = in_channels
    for _ in range(num_convs):
        layers.append(nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        current_in = out_channels
    return nn.Sequential(*layers)


class SegNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.features = nn.Sequential(
            *_make_vgg16_bn_encoder_layers()
        )
        if pretrained:
            self._load_pretrained_vgg16_bn()

    def _load_pretrained_vgg16_bn(self) -> None:
        try:
            weights = VGG16_BN_Weights.IMAGENET1K_V1
            vgg = vgg16_bn(weights=weights)
        except Exception:
            vgg = vgg16_bn(weights=None)
        self.features.load_state_dict(vgg.features.state_dict(), strict=False)

    def forward(self, x: torch.Tensor):
        indices = []
        sizes = []
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                sizes.append(x.size())
                x, idx = layer(x)
                indices.append(idx)
            else:
                x = layer(x)
        return {
            "x": x,
            "indices": indices,
            "sizes": sizes,
        }


class SegNetDecoder(nn.Module):
    def __init__(self, num_class: int = 150):
        super().__init__()
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.dec5 = _conv_block(512, 512, 3)
        self.dec4 = _conv_block(512, 256, 3)
        self.dec3 = _conv_block(256, 128, 3)
        self.dec2 = _conv_block(128, 64, 2)
        self.dec1 = _conv_block(64, 64, 2)
        self.classifier = nn.Conv2d(64, num_class, kernel_size=1)

    def forward(self, encoded, seg_size: Tuple[int, int] = None):
        x = encoded["x"]
        indices = encoded["indices"]
        sizes = encoded["sizes"]

        x = self.unpool5(x, indices[-1], output_size=sizes[-1])
        x = self.dec5(x)
        x = self.unpool4(x, indices[-2], output_size=sizes[-2])
        x = self.dec4(x)
        x = self.unpool3(x, indices[-3], output_size=sizes[-3])
        x = self.dec3(x)
        x = self.unpool2(x, indices[-4], output_size=sizes[-4])
        x = self.dec2(x)
        x = self.unpool1(x, indices[-5], output_size=sizes[-5])
        x = self.dec1(x)
        x = self.classifier(x)

        if seg_size is not None and tuple(x.shape[-2:]) != tuple(seg_size):
            x = F.interpolate(x, size=seg_size, mode="bilinear", align_corners=False)
        return x


class SegNet(nn.Module):
    def __init__(self, num_class: int = 150, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = SegNetEncoder(pretrained=pretrained_encoder)
        self.decoder = SegNetDecoder(num_class=num_class)

    def forward(self, x: torch.Tensor, seg_size: Tuple[int, int] = None):
        encoded = self.encoder(x)
        return self.decoder(encoded, seg_size=seg_size)


def _make_vgg16_bn_encoder_layers() -> List[nn.Module]:
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    return layers
