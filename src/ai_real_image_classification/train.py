import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms 

from model import ResNet
from ai_real_image_classification.data import ai_vs_human_dataset
from ai_real_image_classification.model import ResNet
ROOT_DIR = './ai_vs_human'
NUM_CLASSES = 1
TRAIN_SPLIT_FRAC = 0.8
IMG_SIZE = 224

if __name__ == "__main__":
    img_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = ai_vs_human_dataset(ROOT_DIR, split="train",
                                        transform=img_tf)
    test_ds = ai_vs_human_dataset(ROOT_DIR, split="test",
                                    transform=img_tf)
    
    train_size = int(TRAIN_SPLIT_FRAC * len(train_ds))
    val_size = len(train_ds) - train_size
    train_set, val_set = torch.utils.data.random_split(train_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    model = ResNet(n_class=NUM_CLASSES, pretrained=True, model_name='resnet18')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    save_dir = "./results"

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        save_check=True
    )