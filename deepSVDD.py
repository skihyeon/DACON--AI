import torch.nn as nn
import torch.nn.functional as F
import torch

class deepSVDD(nn.Module):
    def __init__(self, z_dim):
        super(deepSVDD, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)              
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 16, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        
        self.conv3 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn5 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(64*8*8, z_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)                                       
        x = self.pool(F.leaky_relu(self.bn1(x)))                # (b, f, h/2, w/2)
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))                # (b, f, h/4, w/4)

        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))                # (b, f, h/8, w/8)
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn4(x)))                # (b, f, h/16, w/16)
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn5(x)))                # (b, f, h/32, w/32)
        x = x.view(x.size(0), -1)                               # (b, f * h/32 * w/32)
        return self.fc1(x)
    
class C_AE(nn.Module):
    def __init__(self, z_dim):
        super(C_AE, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)              
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 16, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        
        self.conv3 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn5 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(128*8*8, z_dim, bias=False)                      # (f * h/32 * w/32, z)

        # self.deconv1 = nn.ConvTranspose2d(1, 4, 5, bias=False, padding=2)
        # self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        # self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=2)
        # self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        # self.deconv3 = nn.ConvTranspose2d(8, 3, 5, bias=False, padding=2)
        
        self.deconv1 = nn.ConvTranspose2d(8, 64, 5, bias=False, padding=2)
        self.dbn1 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        self.dbn2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(32, 16, 5, bias=False, padding=2)
        self.dbn3 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(16, 8, 5, bias=False, padding=2)
        self.dbn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(8, 3, 5, bias=False, padding=2)


    def encoder(self, x):                                      
        x = self.conv1(x)                                       
        x = self.pool(F.leaky_relu(self.bn1(x)))                # (b, f, h/2, w/2)
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))                # (b, f, h/4, w/4)

        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))                # (b, f, h/8, w/8)
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn4(x)))                # (b, f, h/16, w/16)
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn5(x)))                # (b, f, h/32, w/32)
        x = x.view(x.size(0), -1)                               # (b, f * h/32 * w/32)
        return self.fc1(x)                                      # (b, z)

    def decoder(self, x):
        x = x.view(x.size(0), int(self.z_dim / (8*8)), 8, 8)        # (b, z) -> (b, k, h/32, w/32)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)              # (b, f, h/16, w/16)
        x = self.deconv1(x)                                             
        x = F.interpolate(F.leaky_relu(self.dbn1(x)), scale_factor=2)    # (b, f, h/8, w/8)
        x = self.deconv2(x)                                             
        x = F.interpolate(F.leaky_relu(self.dbn2(x)), scale_factor=2)    # (b, f, h/4, w/4)
        x = self.deconv3(x)                                             
        x = F.interpolate(F.leaky_relu(self.dbn3(x)), scale_factor=2)    # (b, f, h/2, w/2)
        x = self.deconv4(x)
        x = F.interpolate(F.leaky_relu(self.dbn4(x)), scale_factor=2)    # (b, f, h, w)
        x = self.deconv5(x)

        return torch.sigmoid(x)
        

    def forward(self, x):
        # print("x: ",x.shape)
        z = self.encoder(x)
        # print("z: ", z.shape)
        x_hat = self.decoder(z)
        # print("x_hat: ", x_hat.shape)
        return x_hat