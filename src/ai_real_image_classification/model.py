import torch
from torch import nn
from torchvision import models
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
import wandb
import hydra


class Model(LightningModule):
    def __init__(
        self,
        optimizer_cfg,
        lr=1e-3,
        n_class=1,
        p_drop=0.5,
        pretrained=True,
        model_name="resnet18",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer_cfg"])

        self.optimizer_cfg = optimizer_cfg

        model = getattr(models, model_name)(weights="DEFAULT" if pretrained else None)

        # Backbone without the final FC layer
        self.backbone = nn.Sequential(*list(model.children())[:-1])

        # Classification head
        self.fc = nn.Sequential(
            nn.Dropout(p=self.hparams.p_drop), nn.Linear(model.fc.in_features, n_class)
        )

        self.criterium = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def training_step(self, batch):
        data, target = batch

        preds = self(data)
        loss = self.criterium(preds, target.unsqueeze(1).float())

        acc = accuracy(torch.sigmoid(preds).squeeze(1), target, task="binary")

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        if self.logger and hasattr(self.logger, "experiment") and batch_idx == 0:
            # Get the first 8 images from the batch
            images = batch[0][:8]
            preds = torch.sigmoid(self(images)).squeeze(1).tolist()
            y = batch[1][:8].tolist()
            captions = [f"Pred: {p}, Truth: {t}" for p, t in zip(preds[:8], y[:8])]

            self.logger.experiment.log(
                {
                    "val_predictions": [
                        wandb.Image(img, caption=cap)
                        for img, cap in zip(images, captions)
                    ]
                }
            )
        loss, acc = self._shared_eval_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch):
        loss, acc = self._shared_eval_step(batch)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def _shared_eval_step(self, batch):
        x, y = batch
        y_hat = self(x)

        loss = self.criterium(y_hat, y.unsqueeze(1).float())
        acc = accuracy(torch.sigmoid(y_hat).squeeze(1), y, task="binary")
        return loss, acc

    def configure_optimizers(self):
        optimizer_fn = hydra.utils.instantiate(self.optimizer_cfg)
        optimizer = optimizer_fn(self.parameters())
        return optimizer
