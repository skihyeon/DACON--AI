import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder Layers
        self.enc_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_pool = nn.MaxPool2d(2, 2)  # Downsampling
        self.dropout1 = nn.Dropout(0.25)   # Dropout for regularization
        
        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bottleneck_bn = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.25)   # Another Dropout in the bottleneck
        
        # Decoder Layers
        self.dec_convtrans1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_convtrans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(32)
        self.dec_convtrans3 = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(16)
        self.dec_convtrans4 = nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1)
        self.dec_bn4 = nn.BatchNorm2d(3)

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_pool(x)  # Apply pooling after activation
        x = self.dropout1(x)  # Apply dropout
        x = F.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_pool(x)  # Apply pooling again
        x = F.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_pool(x)  # And once more
        # Bottleneck
        x = F.relu(self.bottleneck_bn(self.bottleneck_conv(x)))
        x = self.dropout2(x)  # Dropout in bottleneck
        # Decoder
        x = F.relu(self.dec_bn1(self.dec_convtrans1(x)))
        x = F.interpolate(x, scale_factor=2.0)  # Upsampling
        x = F.relu(self.dec_bn2(self.dec_convtrans2(x)))
        x = F.interpolate(x, scale_factor=2.0)  # Upsampling
        x = F.relu(self.dec_bn3(self.dec_convtrans3(x)))
        x = F.interpolate(x, scale_factor=2.0)  # Upsampling
        x = torch.sigmoid(self.dec_bn4(self.dec_convtrans4(x)))
        return x