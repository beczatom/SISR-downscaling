"""
Based on: Ondřej Podsztavek
Source: https://github.com/podondra/downscaling/
RMSE logging and loss function dependent on input were added.
"""

import lightning
import torchmetrics
import torch


class AbstractModel(lightning.LightningModule):
    """Abstract model for PyTorch Lightning usage"""

    def __init__(self):
        """
        Initialize the model.
        Train and validation metrics (MAE, MSE, RMSE) are initialized.
        """
        super().__init__()
        self.save_hyperparameters()
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "mae": torchmetrics.MeanAbsoluteError(),
                "mse": torchmetrics.MeanSquaredError(),
                "rmse": torchmetrics.MeanSquaredError(squared=False),
            },
            prefix='train/',
        )
        self.val_metrics = self.train_metrics.clone(prefix='val/')
        self.test_metrics = self.train_metrics.clone(prefix='test/')

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Train step.
        Makes step and logs loss and metrics.
        Args:
            batch : torch.Tensor
            batch_idx: int

        Returns:
            loss
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y, x)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_metrics.update(y_hat, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        Makes step and logs loss and metrics.
        Args:
            batch : torch.Tensor
            batch_idx: int

        Returns:
            loss
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y, x)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metrics.update(y_hat, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        Makes step and logs loss and metrics.
        Args:
            batch : torch.Tensor
            batch_idx: int

        Returns:
            loss
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y, x)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metrics.update(y_hat, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """
        Prediction step.
        Args:
            batch : torch.Tensor
            batch_idx : int
            dataloader_idx : int

        Returns:
            output : torch.Tensor
        """
        x, y = batch
        return self(x)

    def on_before_optimizer_step(self, optimizer):
        """
        Called before the optimizer step, logs the gradient norm.
        Args:
            optimizer: torch.optim
        """
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
        self.log("grad_norm", grad_norm, on_step=False, on_epoch=True, prog_bar=True)
