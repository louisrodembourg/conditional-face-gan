import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
import torchvision.transforms as transforms
from PIL import Image
from config import IMAGE_SIZE, ATTR_NAMES, BATCH_SIZE

class CustomCelebA(Dataset):
    def __init__(self, img_dir, attr_path, selected_attrs, transform=None, split='train'):
        self.img_dir = img_dir
        self.transform = transform
        self.selected_attrs = selected_attrs
        
        full_df = pd.read_csv(attr_path)
        
        # Splits: Train (rest) | Val (2000) | Test (2000)
        test_len = 2000
        val_len = 2000
        
        if split == 'train':
            self.attr_df = full_df.iloc[:- (val_len + test_len)]
        elif split == 'val':
            self.attr_df = full_df.iloc[- (val_len + test_len) : - test_len]
        elif split == 'test':
            self.attr_df = full_df.iloc[- test_len :]
        else:
            self.attr_df = full_df

        self.image_names = self.attr_df.iloc[:, 0].values
        self.labels = self.attr_df[selected_attrs].values
        self.labels = (self.labels + 1) // 2 # -1/1 -> 0/1

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, FileNotFoundError) as e:
            warnings.warn(f"Could not read {img_path} ({e}), using next image instead.")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
        attr = torch.tensor(self.labels[idx]).float()
        return image, attr

def get_dataloaders(img_dir, attr_path):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = CustomCelebA(img_dir, attr_path, ATTR_NAMES, transform, 'train')
    val_set = CustomCelebA(img_dir, attr_path, ATTR_NAMES, transform, 'val')
    test_set = CustomCelebA(img_dir, attr_path, ATTR_NAMES, transform, 'test')

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader