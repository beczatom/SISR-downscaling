"""
Based on: Ondřej Podsztavek
Source: https://github.com/podondra/downscaling/
Optimizer scheduler was added.
"""

import torch
import sympy

from .model import AbstractModel


class Upsampler(torch.nn.Sequential):
    """
    Upsampling module of the EDSR model.
    """

    def __init__(self, scale_factor: int, features: int):
        """
        Initialize the Upsampler module.
        Args:
            scale_factor : int  Downscaling factor, how much we need to increase the spatial resolution of the input image
            features : int      Number of channels in the input image
        """
        # get the scale_factor prime factorization
        scale_factors = sympy.factorint(scale_factor, multiple=True)  # multiple = True, to obtain ascending list
        modules = []
        # we will increase the spatial resolution by the prime factors of scale_factor
        for factor in scale_factors:
            modules.append(
                torch.nn.Conv2d(in_channels=features, out_channels=features * (factor ** 2), kernel_size=3, padding=1))
            modules.append(torch.nn.PixelShuffle(factor))
        super().__init__(*modules)


class ResidualBlock(torch.nn.Module):
    """
    Residual block of the EDSR model.
    """

    def __init__(self, features: int):
        """
        Initialize the residual block.
        Args:
            features : int  Number of channels used in the residual block
        """
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.
        Args:
            x : torch.Tensor    Input tensor
        Returns:
            y_hat: torch.Tensor Predicted tensor
        """
        residue = self.block(x)
        residue = residue.mul(0.1)
        residue += x
        return residue


class EDSR(AbstractModel):
    """
    Enhanced Deep Residual Network for Single Image Super-Resolution. Lim et al. 2017.
    """

    def __init__(self, channels: int, scale_factor: int, features: int, residual_blocks: int, loss: torch.nn.Module):
        """
        Initialize the EDSR model.
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
            torch.nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1)
        )
        self.body = torch.nn.Sequential(
            *[ResidualBlock(features=features) for _ in range(residual_blocks)],
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1),
        )
        self.tail = torch.nn.Sequential(
            Upsampler(scale_factor=scale_factor, features=features),
            torch.nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EDSR model.
        Args:
            x : torch.Tensor    Input image
        Returns:
            y_hat: torch.Tensor Predicted image
        """
        x = self.head(x)
        residue = self.body(x)
        residue += x
        x = self.tail(residue)
        return x

    def configure_optimizers(self):
        """Set up the optimizer for the EDSR model."""
        return {
            "optimizer": self.hparams.optimizer,
            "lr_scheduler": {
                "scheduler": self.hparams.lr_scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
