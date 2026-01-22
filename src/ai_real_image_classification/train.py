import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split
import random

from ai_real_image_classification.model import Model
from ai_real_image_classification.data.dataset import AIvsHumanDataset


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    wandb_logger = WandbLogger(project=cfg.project_name, log_model=True)

    # Hydra paths
    orig_cwd = hydra.utils.get_original_cwd()
    data_root = os.path.join(orig_cwd, cfg.data.root_dir)

    # Data setup
    img_tf = transforms.Compose(
        [
            transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
            transforms.ToTensor(),
        ]
    )

    random.seed(cfg.seed)

    dataset = AIvsHumanDataset(data_root, transform=img_tf)

    train_size = int(cfg.data.train_split * len(dataset))
    val_size = int(cfg.data.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.train.batch_size, num_workers=0, shuffle=False
    )

    model = Model(
        n_class=cfg.data.num_classes,
        pretrained=cfg.train.pretrained,
        model_name="resnet18",
        optimizer_cfg=cfg.optimizer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min", filename="best"
    )

    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=10),
        ],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")
    trainer.save_checkpoint("best_model.pth", weights_only=True)


if __name__ == "__main__":
    main()
