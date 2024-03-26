import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm.auto import tqdm

from config import Config
from utils import seed_everything
from data_loader import get_train_loader
from autoencoder import AutoEncoder
from wandb_utils import wandb_init

def train(model, 
          train_loader, 
          criterion, 
          optimizer, 
          num_epochs, 
          device,
          wandb,
          model_save_path):
    model.train()
    best_loss = float('inf')
    best_model = None
    
    for epoch in tqdm(range(num_epochs), total=num_epochs, desc="Epoch"):
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        wandb.log({"epoch": epoch, "loss": avg_loss})
        
        tqdm.write(f"Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_scripted = torch.jit.script(model)
            torch.jit.save(best_model_scripted, model_save_path)
            best_model = best_model_scripted
    
    return best_model

def main():
    config = Config()
    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    wandb = wandb_init(config.project_name, config.entity_name)
    
    train_loader = get_train_loader(config.train_csv, config.batch_size)
    
    autoencoder = AutoEncoder().to(device) 
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(autoencoder.parameters(), lr=config.learning_rate)
    
    best_model = train(autoencoder, 
                       train_loader, 
                       criterion, 
                       optimizer, 
                       config.num_epochs, 
                       device,
                       wandb,
                       model_save_path=config.model_save_path)
    
    wandb.finish()

if __name__ == "__main__":
    main()