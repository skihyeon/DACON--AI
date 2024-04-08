import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from torchvision import transforms

class Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df['img_path'].iloc[idx]
        label = self.df['label'].iloc[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label
    
def crop_by_pixels(img, pixels_top, pixels_bot):
    w, h = img.size
    return img.crop((0, pixels_top, w, h-pixels_bot))

def global_contrast_normalization(x):
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean
    x_scale = torch.mean(torch.abs(x))
    x /= x_scale
    return x


def get_molding_loader(csv_file, batch_size, shuffle=True):
    transform = transforms.Compose([
                 transforms.Resize((256, 512)),
                 transforms.CenterCrop((254, 320)),
                 transforms.Lambda(lambda image: crop_by_pixels(image, 30, 0)), # 224 x 320
                 transforms.ToTensor(),
                 transforms.Lambda(lambda image: global_contrast_normalization(image)),
                 transforms.Lambda(lambda image: TF.adjust_contrast(image, contrast_factor=2)),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ])

    data = Dataset(csv_file, transform=transform)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader

def get_leadframe_loader(csv_file, batch_size, shuffle=True):
    transform = transforms.Compose([
                 transforms.Resize((256, 512)),
                 transforms.CenterCrop((254,320)),
                 transforms.Lambda(lambda image:crop_by_pixels(image, 0, 30)), # 224x320
                 transforms.Grayscale(num_output_channels=1),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.406], std=[0.225])
    ])

    data = Dataset(csv_file, transform=transform)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader