import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from config import Config
from utils import seed_everything, weights_init_normal, wandb_login
from data_loader import get_train_loader
from deepSVDD import deepSVDD, C_AE



class Trainer:
    def __init__(self, config, dataloader, wandb):
        self.config = config
        self.train_loader = dataloader
        self.device = config.general.device
        self.wandb = wandb

    def pretrain(self):
        self.wandb.init(project=self.config.general.project_name_pt, entity=self.config.general.entity_name)

        AE = C_AE(self.config.model.C_AE.latent_dim).to(self.device)
        AE.apply(weights_init_normal)
        optimizer = optim.Adam(AE.parameters(), lr = self.config.model.C_AE.lr,
                               weight_decay=self.config.model.C_AE.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config.model.C_AE.step_size, gamma=self.config.model.C_AE.gamma)

        AE.train()

        loss_func = nn.MSELoss()
        best_loss = float('inf')
        epochs = self.config.model.C_AE.num_epochs
        for epoch in tqdm(range(epochs), total=epochs, desc="Epoch"):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                x_hat = AE(x)
                # reconst_loss = torch.mean(torch.sum((x_hat-x)**2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss = loss_func(x_hat, x)
                reconst_loss.backward()
                optimizer.step()

                total_loss += reconst_loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            self.wandb.log({"epoch": epoch, "loss": avg_loss, "lr": scheduler.get_last_lr()[0] })

            tqdm.write(f'Pretraining Autoencoder... Epoch: {epoch}, Loss: {avg_loss:.3f}')

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = torch.jit.script(AE)
                torch.jit.save(best_model, self.config.model.C_AE.save_model_path)
                # print(f"model saved to {self.config.model.C_AE.save_model_path}")

            scheduler.step()
            
        self.save_weights_for_deepSVDD(AE, self.train_loader)
        self.wandb.finish()

    def save_weights_for_deepSVDD(self, model, dataloader):
        c = self.set_c(model, dataloader)
        net = deepSVDD(self.config.model.C_AE.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, self.config.model.C_AE.save_params_path)
        
    def set_c(self, model, dataloader, eps=0.1):
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encoder(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c
    
    def train(self):
        net = deepSVDD(self.config.model.C_AE.latent_dim).to(self.device)

        if self.config.model.deepSVDD.pretrained == True:
            state_dict = torch.load(self.config.model.deepSVDD.pretrained_path)
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(self.config.model.C_AE.latent_dim).to(self.device)
        
        optimizer = optim.Adam(net.parameters(), lr=self.config.model.deepSVDD.lr,
                               weight_decay=self.config.model.deepSVDD.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config.model.deepSVDD.step_size, gamma = self.config.model.deepSVDD.gamma)
        
        net.train()
        
        self.wandb.init(project=self.config.general.project_name, entity=self.config.general.entity_name)

        best_loss = float('inf')
        epochs = self.config.model.deepSVDD.num_epochs
        # loss_func = nn.MSELoss()
        for epoch in tqdm(range(epochs), total=epochs, desc="Epoch"):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = net(x)
                loss = torch.mean(torch.sum((z-c) ** 2, dim=1))
                # loss = loss_func(z, c)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss/len(self.train_loader) 
            self.wandb.log({"epoch": epoch, "loss": avg_loss, "lr":scheduler.get_last_lr()[0]})
            tqdm.write(f'Training Deep SVDD... Epoch: {epoch}, Loss: {avg_loss:.3f}')
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = torch.jit.script(net)
                torch.jit.save(best_model, self.config.train.model_save_path)
            
            scheduler.step()
            
        self.net = net
        self.c = c


def main():
    config = Config()
    seed_everything(config.general.seed)
    wandb = wandb_login()

    train_loader = get_train_loader(config.train.train_csv, config.train.load_batch_size)

    trainer = Trainer(config, train_loader, wandb)

    if config.model.deepSVDD.pretrained == True:
        trainer.pretrain()
    
    trainer.train()

if __name__ == "__main__":
    main()