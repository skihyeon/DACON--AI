import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from configs.config import Config
from utils.utils import seed_everything, weights_init_normal, wandb_login, update_wandb_config
from dataloaders.dataloader import get_molding_loader, get_leadframe_loader
from models.deepSVDD_split import net_molding, net_leadframe, AE_molding, AE_leadframe

class Trainer:
    def __init__(self, train_mode, config, dataloader, wandb=None):
        if train_mode not in ['molding', 'leadframe']:
            raise ValueError("train_mode must be 'molding' or 'leadframe'")
        self.mode = train_mode
        self.config = config
        self.train_loader = dataloader
        self.wandb = wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def pretrain(self):
        config = self.config.pretrain_molding if self.mode=='molding' else self.config.pretrain_leadframe

        AE = AE_molding(config.z_dim) if self.mode=='molding' else AE_leadframe(config.z_dim)
        AE.to(self.device)
        AE.apply(weights_init_normal)

        optimizer = optim.Adam(AE.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=config.step_size, gamma=config.gamma)

        AE.train()

        best_loss = float('inf')
        num_epochs = config.epochs
        loss_func = nn.MSELoss()

        with tqdm(range(num_epochs), total=num_epochs, desc="Epoch") as pbar:
            for epoch in range(num_epochs):
                total_loss = 0.0
                for imgs, _ in self.train_loader:
                    imgs = imgs.float().to(self.device)

                    optimizer.zero_grad()
                    reconstructed = AE(imgs)
                    loss = torch.mean(torch.sum((reconstructed - imgs) ** 2, dim=tuple(range(1, reconstructed.dim()))))
                    # loss = loss_func(reconstructed, imgs)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                avg_loss = total_loss / len(self.train_loader)
                if self.wandb != None:
                    self.wandb.log({f'{config.run_name}/epoch': epoch, f'{config.run_name}/loss': avg_loss})
                pbar.set_postfix_str(f'{config.run_name}... Epoch: {epoch}, Loss: {avg_loss:.3f}')
                pbar.update()
                scheduler.step()

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model = torch.jit.script(AE)
                    os.makedirs(config.save_path, exist_ok=True)
                    torch.jit.save(best_model, config.save_model_path)
            
            self.save_weights_for_deepSVDD(AE, self.train_loader)
        

    def save_weights_for_deepSVDD(self, model, dataloader):
        config = self.config.train_molding if self.mode=='molding' else self.config.train_leadframe
        c = self.set_c(model, dataloader)
        net = net_molding(config.z_dim) if self.mode=='molding' else net_leadframe(config.z_dim)
        net.to(self.device)

        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)

        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, config.param_path)

    def set_c(self, model, dataloader, eps=0.1):
        model.eval()
        z_ = []
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.float().to(self.device)
                z = model.encoder(imgs)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c    

    def train(self):
        config = self.config.train_molding if self.mode =='molding' else self.config.train_leadframe
        net = net_molding(config.z_dim) if self.mode=='molding' else net_leadframe(config.z_dim)
        net.to(self.device)

        if config.pretrained == True:
            state_dict = torch.load(config.param_path)
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(config.z_dim).to(self.device)

        optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

        net.train()

        best_loss = float('inf')
        num_epochs = config.epochs
        loss_func = nn.MSELoss()
        with tqdm(range(num_epochs), total=num_epochs, desc="Epoch") as pbar:
            for epoch in range(num_epochs):
                total_loss = 0.0
                for imgs, _ in self.train_loader:
                    imgs = imgs.float().to(self.device)

                    optimizer.zero_grad()
                    z = net(imgs)
                    loss = torch.mean(torch.sum((z-c)**2, dim=1))
                    # loss = loss_func(z,c)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                
                avg_loss = total_loss/len(self.train_loader)
                if self.wandb != None:
                    self.wandb.log({f'{config.run_name}epoch': epoch, f'{config.run_name}/loss': avg_loss})
                pbar.set_postfix_str(f'{config.run_name}... Epoch: {epoch}, Loss: {avg_loss:.3f}')
                pbar.update()
                scheduler.step()

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model = torch.jit.script(net)
                    torch.jit.save(best_model, config.save_model_path)
                
                self.net = net
                self.c = c

def train_model(mode, config, wandb):
    if mode == 'molding':
        loader_function = get_molding_loader
        mode_config = config.train_molding
    else:
        loader_function = get_leadframe_loader
        mode_config = config.train_leadframe

    train_loader = loader_function(mode_config.img_path, mode_config.batch_size)
    trainer = Trainer(mode, config, train_loader, wandb)

    if mode_config.pretrained:
        trainer.pretrain()
    trainer.train()

    del train_loader, trainer

def main():
    config = Config()
    seed_everything(123)

    wandb = wandb_login()
    wandb.init(project=config.wandb.project_name, entity=config.wandb.entity, reinit=True, name=config.wandb.run_name)
    update_wandb_config(wandb, config)

    
    for mode in ['molding', 'leadframe']:
        train_model(mode, config, wandb)
    wandb.finish()


if __name__ == "__main__":
    main()