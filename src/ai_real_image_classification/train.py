import os
import torch
import random
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from ai_real_image_classification.model import Model
from ai_real_image_classification.data.dataset import AIvsHumanDataset


def get_data_path(cfg: DictConfig):
    """Determines the data root based on the environment."""
    if cfg.gcp.use_gcp:
        return f"/gcs/{cfg.gcp.bucket_name}/{cfg.gcp.data_dir}"
    else:
        # Local run
        orig_cwd = hydra.utils.get_original_cwd()
        return os.path.join(orig_cwd, cfg.data.root_dir)


def get_output_dir(cfg: DictConfig):
    """Determines where to save models/logs."""
    if cfg.gcp.use_gcp:
        return f"/gcs/{cfg.gcp.bucket_name}/{cfg.gcp.output_dir}"
    return "./models"


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Wandb logger
    wandb_logger = WandbLogger(project=cfg.project_name, log_model=True)

    # Setup Paths
    data_root = get_data_path(cfg)
    output_dir = get_output_dir(cfg)

    print(f"--- Running with data_root: {data_root} ---")

    # Data setup
    img_tf = transforms.Compose(
        [
            transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
            transforms.ToTensor(),
        ]
    )

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset = AIvsHumanDataset(data_root, transform=img_tf)

    # Split dataset
    train_size = int(cfg.data.train_split * len(dataset))
    val_size = int(cfg.data.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Data Loaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        persistent_workers=(cfg.train.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        persistent_workers=(cfg.train.num_workers > 0),
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.train.batch_size, num_workers=0, shuffle=False
    )

    # Model Setup
    model = Model(
        n_class=cfg.data.num_classes,
        pretrained=cfg.train.pretrained,
        model_name="resnet18",
    )

    # Callbacks & Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        monitor="val_loss",
        mode="min",
        filename="best-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices="cuda" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=10),
        ],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")

    final_path = os.path.join(output_dir, "final_model.pth")
    trainer.save_checkpoint(final_path)


if __name__ == "__main__":
    main()
