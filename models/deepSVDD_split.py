import torch
import torch.nn as nn
import torch.nn.functional as F

class net_molding(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(128*28*40, z_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
    

class AE_molding(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        nn.init.xavier_uniform_(self.conv3.weight)

        self.fc1 = nn.Linear(128*28*40, z_dim, bias=False)
        self.fbn1 = nn.BatchNorm1d(z_dim, eps=1e-04, affine=False)

        self.deconv1 = nn.ConvTranspose2d(int(z_dim/(28*40)), 128, 5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.dbn1 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1, bias=False)
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.dbn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1, bias=False)
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.dbn3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, stride=2, padding=2, output_padding=1, bias=False)
        nn.init.xavier_uniform_(self.deconv4.weight)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        return self.fbn1(self.fc1(x))
    
    def decoder(self, x):
        x = x.view(x.size(0), int(self.z_dim/ (28*40)), 28, 40)
        x = F.elu(x)
        x = self.deconv1(x)
        x = F.elu(self.dbn1(x))
        x = self.deconv2(x)
        x = F.elu(self.dbn2(x))
        x = self.deconv3(x)
        x = F.elu(self.dbn3(x))
        x = self.deconv4(x)
        return torch.sigmoid(x)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class net_leadframe(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 32, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(128*28*40, z_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
    

class AE_leadframe(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 32, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        nn.init.xavier_uniform_(self.conv3.weight)

        self.fc1 = nn.Linear(128*28*40, z_dim, bias=False)
        self.fbn1 = nn.BatchNorm1d(z_dim, eps=1e-04, affine=False)

        self.deconv1 = nn.ConvTranspose2d(int(z_dim/(28*40)), 128, 5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.dbn1 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1, bias=False)
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.dbn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1, bias=False)
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.dbn3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 1, 5, stride=2, padding=2, output_padding=1, bias=False)
        nn.init.xavier_uniform_(self.deconv4.weight)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        return self.fbn1(self.fc1(x))
    
    def decoder(self, x):
        x = x.view(x.size(0), int(self.z_dim/ (28*40)), 28, 40)
        x = F.elu(x)
        x = self.deconv1(x)
        x = F.elu(self.dbn1(x))
        x = self.deconv2(x)
        x = F.elu(self.dbn2(x))
        x = self.deconv3(x)
        x = F.elu(self.dbn3(x))
        x = self.deconv4(x)
        return torch.sigmoid(x)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat