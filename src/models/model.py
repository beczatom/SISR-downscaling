
import lightning
import torchmetrics

class AbstractModel(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "mae" : torchmetrics.MeanAbsoluteError(),
                "mse" : torchmetrics.MeanSquaredError(),
                "rmse" : torchmetrics.MeanSquaredError(squared=False),
            },
            prefix = 'train/',
        )
        self.val_metrics = self.train_metrics.clone(prefix='val/')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log('hp_metric', metrics['val/rmse'])