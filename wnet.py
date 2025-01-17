import torch
import torch.nn as nn
import torch.nn.functional as F

# gets centered crop of the fiven tensor
# image dimmensions = 1024 x 2048
# torch.Size([1, 1, 32, 32])
# size (images, features, height, width)
def tensorCenterCrop(tensor, height, width):
    return tensor
    heightStartIdx = ((tensor.size()[2] +1) - height) / 2
    widthStartIdx = ((tensor.size()[3] +1) - width) / 2
    return tensor[:,:,int(heightStartIdx):int(heightStartIdx+height), int(widthStartIdx):int(widthStartIdx+width)]

# torch.cat((first_tensor, second_tensor), 0)

class WNET(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        self.conv1_1   = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2   = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1 )
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0, return_indices=False, ceil_mode=False)

        self.conv2_1   = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1 )
        self.conv2_2   = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1 )
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0, return_indices=False, ceil_mode=False)

        
        self.conv3_1   = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1 )
        self.conv3_2   = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1 )
        self.deconv3 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2, padding=0, output_padding=0)
        
        self.conv4_1   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1 )
        self.conv4_2   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1 )
        self.deconv4 = nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        
        
        self.conv5_1   = nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1)
        self.conv5_2   = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1 )
        self.pool5 = nn.MaxPool2d(2, stride=2, padding=0, return_indices=False, ceil_mode=False)

        self.conv6_1   = nn.Conv2d(224, 64, kernel_size=3, stride=1, padding=1 )
        self.conv6_2   = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1 )
        self.pool6 = nn.MaxPool2d(2, stride=2, padding=0, return_indices=False, ceil_mode=False)

        
        self.conv7_1   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1 )
        self.conv7_2   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1 )
        self.deconv7 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, output_padding=0)

        self.conv8_1   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1 )
        self.conv8_2   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1 )
        self.deconv8 = nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        
        self.conv9_1   = nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1 )
        self.conv9_2   = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1 )
        
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1, stride=1, padding=0, )
  

    def forward(self, x):
        
        outConv1 = F.relu(self.conv1_1(x))
#         outConv1 = F.relu(self.conv1_2(outConv1))
        out1 = self.pool1(outConv1)
    
        outConv2 = F.relu(self.conv2_1(out1))
#         outConv2 = F.relu(self.conv2_2(outConv2))
        out2 = self.pool2(outConv2)
        
        outConv3 = F.relu(self.conv3_1(out2))
#         outConv6 = F.relu(self.conv6_2(outConv6))
        out3 = self.deconv3(outConv3)
        
        outConv4 = F.relu(self.conv4_1(torch.cat((out3, tensorCenterCrop(outConv2, out3.size()[2], out3.size(3))), 1)))
#         outConv7 = F.relu(self.conv7_2(outConv7))
        out4 = self.deconv4(outConv4)
        
        outConv5 = F.relu(self.conv5_1(torch.cat((out4, tensorCenterCrop(outConv1, out4.size()[2], out4.size(3))), 1)))
#         outConv1 = F.relu(self.conv1_2(outConv1))
        out5 = self.pool5(outConv5)
        
        outConv6 = F.relu(self.conv6_1(torch.cat((out5, tensorCenterCrop(outConv4, out5.size()[2], out5.size(3))), 1)))
#         outConv2 = F.relu(self.conv2_2(outConv2))
        out6 = self.pool6(outConv6)
        
        outConv7 = F.relu(self.conv7_1(torch.cat((out6, tensorCenterCrop(outConv3, out5.size()[2], out6.size(3))), 1)))
#         outConv6 = F.relu(self.conv6_2(outConv6))
        out7 = self.deconv7(outConv7)

        outConv8 = F.relu(self.conv8_1(torch.cat((out7, tensorCenterCrop(outConv6, out7.size()[2], out7.size(3))), 1)))
#         outConv7 = F.relu(self.conv7_2(outConv7))
        out8 = self.deconv8(outConv8)
                  
        outConv9 = F.relu(self.conv9_1(torch.cat((out8, tensorCenterCrop(outConv5, out8.size()[2], out8.size(3))), 1)))
#         outConv7 = F.relu(self.conv7_2(outConv7))
#         out9 = self.deconv9(outConv9)
         
        preds = self.classifier(outConv9)
  
        return preds  # size=(N, n_class, x.H/1, x.W/1)
