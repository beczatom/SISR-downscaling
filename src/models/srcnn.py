"""
Based on: Paula Harder
Source: https://github.com/RolnickLab/constrained-downscaling/
"""

import torch
import sympy

from .model import AbstractModel


class Upsampler(torch.nn.Sequential):
    """
    Upsampling module of the SR-CNN model.
    """

    def __init__(self, scale_factor: int, features: int, method: str = 'shuffle'):
        """
        Initialize the Upsampler module.
        Args:
            scale_factor : int  Downscaling factor, how much we need to increase the spatial resolution of the input image
            features : int      Number of channels in the input image
            method : str        Upsampling method, choose between 'shuffle' and 'transpose'
        """
        # get the scale_factor prime factorization
        scale_factors = sympy.factorint(scale_factor, multiple=True)  # multiple = True, to obtain ascending list
        modules = []
        # we will increase the spatial resolution by the prime factors of scale_factor
        for factor in scale_factors:
            if method == 'shuffle':
                modules.append(
                    torch.nn.Conv2d(in_channels=features, out_channels=features * (factor ** 2), kernel_size=3,
                                    padding=1))
                modules.append(torch.nn.PixelShuffle(factor))
            elif method == 'transpose':
                modules.append(torch.nn.ConvTranspose2d(in_channels=features, out_channels=features, kernel_size=factor,
                                                        stride=factor, padding=0))

        super().__init__(*modules)


class ResidualBlock(torch.nn.Module):
    """
    Residual block of the SR-CNN model.
    """

    def __init__(self, features: int):
        """
        Initialize the residual block.
        Args:
            features : int  Number of channels used in the residual block
        """
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.
        Args:
            x : torch.Tensor    Input tensor
        Returns:
            y_hat: torch.Tensor Predicted tensor
        """
        residue = self.block(x)
        residue += x
        return self.relu(residue)


class SRCNN(AbstractModel):
    """
    Image Super-Resolution Using Deep Convolutional Networks. Dong et al. 2016.
    """

    def __init__(self, channels: int, scale_factor: int, features: int, residual_blocks: int, loss: torch.nn.Module,
                 upsample_method: str = 'shuffle'):
        """
        Initialize the SR-CNN model.
        Args:
            channels : int          Number of channels in the input image
            scale_factor : int      Downscaling factor
            features : int          Number of channels in residual blocks
            residual_blocks : int   Number of residual blocks
            loss : torch.nn.Module  Loss function
        """
        super().__init__()
        self.loss = loss

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )
        self.upsampler = Upsampler(scale_factor=scale_factor, features=features, method=upsample_method)
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            *[ResidualBlock(features=features) for _ in range(residual_blocks)],
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )
        self.tail = torch.nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SR-CNN model.
        Args:
            x : torch.Tensor    Input image
        Returns:
            y_hat: torch.Tensor Predicted image
        """
        out = self.head(x)
        out = self.upsampler(out)
        out = self.body(out)
        out = self.tail(out)
        return out
