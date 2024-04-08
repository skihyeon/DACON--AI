import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from PIL import Image

class Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform(transform)

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

