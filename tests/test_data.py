import os.path
import pytest
import torch
import torchvision

from ai_real_image_classification.data.dataset import AIvsHumanDataset


@pytest.mark.skipif(not os.path.exists("./tests"), reason="No dataset found")
def test_data():
    data_dir = "./tests"
    dataset = AIvsHumanDataset(
        root_dir=data_dir, transform=torchvision.transforms.ToTensor()
    )
    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"
    # Check a sample
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor), "Image is not a tensor"
    assert isinstance(label, int), "Label is not an integer"
    assert img.shape[0] == 3, "Image does not have 3 channels"


@pytest.mark.skipif(not os.path.exists("./tests"), reason="No dataset found")
def test_transforms():
    data_dir = "./tests"
    dataset = AIvsHumanDataset(
        root_dir=data_dir,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        ),
    )
    sample, target = dataset[0]

    # Check shape after transform
    assert sample.shape == (3, 224, 224), f"Unexpected shape: {sample.shape}"
    assert isinstance(sample, torch.Tensor), "Transformed sample is not a tensor"
    assert isinstance(target, int), "Target is not an integer"
