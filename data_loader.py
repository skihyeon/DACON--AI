from torch.utils.data import DataLoader
from torchvision import transforms
from load_dataset import CustomDataset

def get_train_loader(csv_file, batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = CustomDataset(csv_file=csv_file, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    return train_loader

def get_test_loader(csv_file, batch_size, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_data = CustomDataset(csv_file=csv_file, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    return test_loader