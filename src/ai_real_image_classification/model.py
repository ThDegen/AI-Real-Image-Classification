import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from pytorch_lightning import LightningModule, Trainer
from torchmetrics.functional import accuracy

class Model(LightningModule):
    def __init__(self, lr=1e-3, n_class=1, p_drop=0.5, pretrained=True, model_name='resnet18') -> None:
        super().__init__()
        self.save_hyperparameters()

        model = getattr(models, model_name)(weights='DEFAULT' if pretrained else None)

        # Backbone without the final FC layer
        self.backbone = nn.Sequential(*list(model.children())[:-1])

        # Classification head
        self.fc = nn.Sequential(
            nn.Dropout(p=self.hparams.p_drop),
            nn.Linear(model.fc.in_features, n_class)
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
        
        acc = accuracy(
                    torch.sigmoid(preds).squeeze(1), 
                    target, 
                    task='binary'
                )
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch) -> None:
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
        acc = accuracy(
                    torch.sigmoid(y_hat).squeeze(1), 
                    y, 
                    task='binary'
                )
        return loss, acc
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)