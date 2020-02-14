import torch.nn as nn
import torchvision


class VGG(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        # VGG11 output architecture:
        # conv3-64-1
        # max 2x2-2
        # conv3-128-1
        # max 2x2-2
        # conv3-256-1
        # conv3-256-1
        # max 2x2-2
        # conv3-512-1
        # conv3-512-1
        # max 2x2-2
        # conv3-512-1
        # conv3-512-1
        # max 2x2-2
        # --------
        # FC-4096
        # FC-4096
        # FC-1000
        # soft-max
        mod = torchvision.models.vgg11_bn(pretrained=True)
        # take only the feature portion of the model (no avg pool or classification)
        self.mod = mod.features
        # freeze pre-trained layers
        for param in self.mod.parameters():
            param.requires_grad = False
        self.n_class = n_class
        #self.fc = nn.Linear(1000, 512)
        #self.bnd1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        
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

    def forward(self, x):
        out_encoder = self.mod

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
        
        encoded = out_encoder(x)
        decoded = out_decoder(encoded)

        score = self.classifier(decoded)

        return score  # size=(N, n_class, x.H/1, x.W/1)