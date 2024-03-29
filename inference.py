import torch.nn as nn
import torch
import numpy as np

from config import Config
from data_loader import get_test_loader


config = Config()

device = config.device()
test_loader = get_test_loader(config.test_csv, config.batch_size, shuffle=False)
model = torch.jit.load(config.model_save_path).to(device)
model.eval()

reconstruction_errors = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        reconstructed = model(images)

        loss = nn.MSELoss(reduction='none')
        reconstruction_error = loss(reconstructed, images)
        reconstruction_error = torch.mean(reconstruction_error, [1, 2, 3]) 
        reconstruction_errors.extend(reconstruction_error.cpu().numpy())

reconstruction_errors = np.array(reconstruction_errors)

mean_error = np.mean(reconstruction_errors)
std_error = np.std(reconstruction_errors)
threshold = mean_error + std_error * 0.7 

labels = (reconstruction_errors > threshold).astype(int)
print("labels:", labels)
print("임계값:", threshold)
print("라벨링된 이미지 개수:", len(labels))
print("정상 이미지 개수:", np.sum(labels == 0))
print("비정상 이미지 개수:", np.sum(labels == 1))