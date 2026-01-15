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

        self.df = pd.read_csv(f'{root_dir}/{split}.csv')
        
        self.img_dirs = sorted(glob(os.path.join(root_dir, folder, '*')))
        self.labels = []

        print("Indexing labels...")
        label_map = dict(zip(self.df['file_name'], self.df['label']))
        
        # Assign labels
        for v in tqdm(self.img_dirs, desc="Assigning labels"):
            file_name = os.path.basename(v)
            
            label = label_map.get("train_data/"+ file_name, -1)
            self.labels.append(label)

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        img_path = self.img_dirs[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

if __name__ == "__main__":
    print("Testing ai_vs_human_dataset...")
    dataset = ai_vs_human_dataset(root_dir='./data', split='train')
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        img, label = dataset[5]
        print(f"Image shape: {img.size}, Label: {label}")