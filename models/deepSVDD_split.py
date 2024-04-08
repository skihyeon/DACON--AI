import torch
import torch.nn as nn
import torch.nn.functional as F

class net_molding(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(4*50*70, z_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
    

class AE_molding(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4*50*70, z_dim, bias=False)

        self.fc2 = nn.Linear(z_dim, 4*50*70, bias=False)
        self.deconv1 = nn.ConvTranspose2d(4, 8, 5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(8, 3, 5, stride=2, padding=2, output_padding=1, bias=False)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
    
    def decoder(self, x):
        x = self.fc2(x)
        x = x.view(x.size(0), 4, 50, 70)
        x = F.leaky_relu(self.bn3(self.deconv1(x)))
        x = self.deconv2(x)
        return torch.sigmoid(x)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class net_leadframe(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2) # GrayScale
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(4*51*75, z_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
    

class AE_leadframe(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4*51*75, z_dim, bias=False)

        self.fc2 = nn.Linear(z_dim, 4*51*75, bias=False)
        self.deconv1 = nn.ConvTranspose2d(4, 8, 5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(8, 3, 5, stride=2, padding=2, output_padding=1, bias=False)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
    
    def decoder(self, x):
        x = self.fc2(x)
        x = x.view(x.size(0), 4, 51, 75)
        x = F.leaky_relu(self.bn3(self.deconv1(x)))
        x = self.deconv2(x)
        return torch.sigmoid(x)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat