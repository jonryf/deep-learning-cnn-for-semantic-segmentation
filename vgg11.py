import torch.nn as nn
import torchvision


class RESNET(nn.Module):

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
        self.fc = nn.Linear(1000, 512)
        self.bnd1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        # output
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.ReLU = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.ReLU
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        # self.ReLU
        self.bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv3 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.ReLU
        self.bn4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        # self.ReLU
        self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv5 = nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #self.ReLU
        self.bn6 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv6 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        #self.ReLU
        self.bn7 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv7 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        #self.ReLU
        self.bn8 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv8 = nn.ConvTranspose2d(64, self.n_class, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


    def forward(self, x):
        out_encoder = nn.Sequential(
            self.mod,
        )

        out_decoder = nn.Sequential(
            # output
            self.unpool1,
            self.ReLU,
            self.bn1,
            self.deconv1,
            # self.ReLU
            self.bn2,
            self.deconv2,
            self.unpool2,
            # self.ReLU
            self.bn3,
            self.deconv3,
            # self.ReLU
            self.bn4,
            self.deconv4,
            self.unpool3,
            # self.ReLU
            self.bn5,
            self.deconv5,
            # self.ReLU
            self.bn6,
            self.deconv6,
            self.unpool4,
            # self.ReLU
            self.bn7,
            self.deconv7,
            self.unpool5,
            # self.ReLU
            self.bn8,
            self.deconv8,
        )

        encoded = out_encoder(x)
        decoded = out_decoder(encoded)

        score = self.classifier(decoded)

        return score  # size=(N, n_class, x.H/1, x.W/1)