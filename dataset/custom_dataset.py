import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class TreeCanopyDataset(Dataset):
    def __init__(self, rgb_dir, fake_dir, label_dir, transform=None):
        self.rgb_paths = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.tif')])
        self.fake_paths = sorted([os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.tif')])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')])

        assert len(self.rgb_paths) == len(self.fake_paths) == len(self.label_paths), "데이터 수 불일치"
        self.transform = transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        fake = Image.open(self.fake_paths[idx]).convert('RGB')  # Assume 3-channel NIR-R-G as RGB
        label = Image.open(self.label_paths[idx]).convert('L')  # Binary mask

        # Transform (공통 처리)
        if self.transform:
            rgb = self.transform(rgb)
            fake = self.transform(fake)
            label = self.transform(label)

        return {
            'rgb': rgb,            # B × 3 × 512 × 512
            'fake': fake,          # B × 3 × 512 × 512
            'label': label         # B × 1 × 512 × 512
        }
