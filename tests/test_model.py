import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from ai_real_image_classification.model import Model  
import warnings


def test_model():
    """
    Runs a single batch of training, validation, and testing 
    to ensure all model methods (steps, logging, optimizers) are reachable.
    """
    warnings.filterwarnings("ignore")
    model = Model()

    data = torch.randn(10, 3, 224, 224)
    targets = torch.randint(0, 2, (10,))
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=2)

    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",  
        devices=1,
        logger=False        
    )

    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
    trainer.test(model, dataloaders=dataloader)