import os
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import pandas as pd
from tqdm import tqdm

class ai_vs_human_dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split == 'train':
            folder = "train_data"
        elif split == 'test':
            folder = "test_data_v2"
        else:
            raise ValueError("split must be 'train' or 'test'")

        # Read csv
        self.df = pd.read_csv(f'{root_dir}/{split}.csv')
        self.img_dirs = sorted(glob(os.path.join(root_dir, folder, '*')))
        self.labels = []

        # Assign labels based on csv
        for v in tqdm(self.img_dirs, desc="Assigning labels", total=len(self.img_dirs)):
            file_name = os.path.basename(v)
            meta = self.df.loc[self.df['file_name'] == file_name]
            if not meta.empty:
                self.labels.append(meta['label'].item())
            else:
                self.labels.append(-1)  

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        img_dir = self.img_dirs[idx]
        label = self.labels[idx]
        img = Image.open(os.path.join(img_dir, '*.jpg')).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
    
if __name__ == "__main__":
    print("Testing ai_vs_human_dataset...")
    dataset = ai_vs_human_dataset(root_dir='./', split='train')
    print(f"Dataset size: {len(dataset)}")
    img, label = dataset[0]
    print(f"Image shape: {img.size}, Label: {label}")