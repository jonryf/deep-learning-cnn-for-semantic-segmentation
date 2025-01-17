import torch.nn as nn
import torch

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        out_encoder = nn.Sequential(
            self.conv1,
            self.bnd1,
            self.relu,
            self.conv2,
            self.bnd2,
            self.relu,
            self.conv3,
            self.bnd3,
            self.relu,
            self.conv4,
            self.bnd4,
            self.relu,
            self.conv5,
            self.bnd5,
            self.relu,

        )

        out_decoder = nn.Sequential(
            self.deconv1,
            self.bn1,
            self.relu,
            self.deconv2,
            self.bn2,
            self.relu,
            self.deconv3,
            self.bn3,
            self.relu,
            self.deconv4,
            self.bn4,
            self.relu,
            self.deconv5,
            self.bn5,
            self.relu
        )
        torch.cuda.empty_cache()

        return self.classifier(out_decoder(out_encoder(x)))  # size=(N, n_class, x.H/1, x.W/1)