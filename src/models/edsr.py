
import torch
import sympy

from .model import AbstractModel

class Upsampler(torch.nn.Sequential):
    def __init__(self, scale_factor : int, features : int):
        scale_factors = sympy.factorint(scale_factor, multiple=True) # multiple = True, to obtain ascending list
        modules = []
        for factor in scale_factors:
            modules.append(torch.nn.Conv2d(in_channels=features, out_channels=features * (factor ** 2), kernel_size=3, padding=1))
            modules.append(torch.nn.PixelShuffle(factor))
        super().__init__(*modules)


class ResidualBlock(torch.nn.Module):
    def __init__(self, features : int):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1),
        )

    def forward(self, x):
        residue = self.block(x)
        residue = residue.mul(0.1)
        residue += x
        return residue


class EDSR(AbstractModel):
    def __init__(self, channels : int, scale_factor : int):
        super().__init__()
        self.loss = torch.nn.L1Loss()
        features = 32
        residual_blocks = 2

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(channels, features, kernel_size=3, padding=1)
        )
        self.body = torch.nn.Sequential(
            *[ResidualBlock(features=features) for _ in range(residual_blocks)],
            torch.nn.Conv2d(features, features, kernel_size=3, padding=1),
        )
        self.tail = torch.nn.Sequential(
            Upsampler(scale_factor=scale_factor, features=features),
            torch.nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=3, padding=1),
        )



    def forward(self, x):
        x = self.head(x)
        residue = self.body(x)
        residue += x
        x = self.tail(residue)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)