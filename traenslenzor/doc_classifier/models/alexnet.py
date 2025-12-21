import torch
from jaxtyping import Float
from pydantic import Field
from torch import nn

from ..utils import BaseConfig


class AlexNetParams(BaseConfig["AlexNet"]):
    """Parameter configuration for AlexNet."""

    target: type["AlexNet"] = Field(default_factory=lambda: AlexNet, exclude=True)
    num_classes: int
    dropout: float = 0.5
    in_channels: int = 1
    """Number of input channels. Set to 1 for grayscale; set to 3 for RGB."""


# B = batch, C = channels, H = height, W = width, N = num_classes
ImageBatch = Float[torch.Tensor, "B C H W"]
Logits = Float[torch.Tensor, "B N"]


class AlexNet(nn.Module):
    """Implementation of the AlexNet model as introduced by Krizhevsky et al.

    # https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    # https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py

    ----------------------------------------------------------------
    Layer (type)            Output Shape                     Param #
    ================================================================
    Conv2d-1                [-1, 96, 55, 55]                  34,944
    ReLU-2                  [-1, 96, 55, 55]                       0
    MaxPool2d-3             [-1, 96, 27, 27]                       0
    Conv2d-4                [-1, 256, 27, 27]                614,656
    ReLU-5                  [-1, 256, 27, 27]                      0
    MaxPool2d-6             [-1, 256, 13, 13]                      0
    Conv2d-7                [-1, 384, 13, 13]                885,120
    ReLU-8                  [-1, 384, 13, 13]                      0
    Conv2d-9                [-1, 384, 13, 13]              1,327,488
    ReLU-10                 [-1, 384, 13, 13]                      0
    Conv2d-11               [-1, 256, 13, 13]                884,992
    ReLU-12                 [-1, 256, 13, 13]                      0
    MaxPool2d-13            [-1, 256, 6, 6]                        0
    AdaptiveAvgPool2d-14    [-1, 256, 6, 6]                        0
    Dropout-15              [-1, 9216]                             0
    Linear-16               [-1, 4096]                    37,752,832
    ReLU-17                 [-1, 4096]                             0
    Dropout-18              [-1, 4096]                             0
    Linear-19               [-1, 4096]                    16,781,312
    ReLU-20                 [-1, 4096]                             0
    Linear-21               [-1, 10]                          40,970
    ================================================================
    Total params: 58,322,314
    Trainable params: 58,322,314
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 11.15
    Params size (MB): 222.48
    Estimated Total Size (MB): 234.21
    ----------------------------------------------------------------
    """

    def __init__(self, params: AlexNetParams):
        super().__init__()
        self.params = params

        self.features = nn.Sequential(
            nn.Conv2d(self.params.in_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.params.dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.params.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.params.num_classes),
        )

    def forward(self, x: ImageBatch) -> Logits:
        """Forward pass through AlexNet.

        Args:
            x: Input tensor with shape (B, C, H, W) where:
               B = batch size, C = ``in_channels`` (default 1 for grayscale), H = height, W = width.

        Returns:
            Logits with shape (B, N) where N = num_classes.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from torchsummary import summary

    model = AlexNetParams(num_classes=10).setup_target()
    summary(model, (3, 224, 224), device="cpu")
