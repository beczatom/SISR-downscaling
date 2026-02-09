import torch
import lightning.pytorch.cli

import datasets
import models
import loss

torch.set_float32_matmul_precision('high')


def cli():
    lightning.pytorch.cli.LightningCLI(seed_everything_default=42)


if __name__ == "__main__":
    cli()
