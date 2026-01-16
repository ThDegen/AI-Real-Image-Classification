import os.path
import pytest
import torch
import torchvision

from ai_real_image_classification.data.dataset import AIvsHumanDataset

FULL_DATA_PATH = "./data"
TEST_DATA_PATH = "./tests"

def get_dataset_config():
    """
    Returns the path and expected count based on availability of datasets.
    """
    if os.path.exists(FULL_DATA_PATH):
        return FULL_DATA_PATH, 79950
    elif os.path.exists(TEST_DATA_PATH):
        return TEST_DATA_PATH, 2 
    else:
        return None, 0

@pytest.mark.skipif(get_dataset_config()[0] is None, reason="No dataset found")
def test_data():
    data_dir, total = get_dataset_config()
    dataset = AIvsHumanDataset(root_dir=data_dir, transform=torchvision.transforms.ToTensor())
    assert len(dataset) == total, f"Expected dataset length {total}, got {len(dataset)}"

    # Check a sample
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor), "Image is not a tensor"
    assert isinstance(label, int), "Label is not an integer"
    assert img.shape[0] == 3, "Image does not have 3 channels"