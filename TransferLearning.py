#!/usr/bin/env python
# coding: utf-8

# In[37]:


from torchvision import utils
from vgg11 import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.modules.loss as loss
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

# In[38]:


train_dataset = CityScapesDataset(csv_file='train.csv')
val_dataset = CityScapesDataset(csv_file='val.csv')
test_dataset = CityScapesDataset(csv_file='test.csv')
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=1,
                        num_workers=1,
                        shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         num_workers=1,
                         shuffle=True)


# In[39]:
epochs = 2
# criterion = # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
criterion = loss.CrossEntropyLoss()
resnet_model = RESNET(n_class=n_class)
# fcn_model.apply(init_weights)
# fcn_model = torch.load('best_model')
optimizer = optim.Adam(resnet_model.parameters(), lr=5e-3)

# In[36]:

use_gpu = torch.cuda.is_available()
if use_gpu:
    fcn_model = resnet_model.cuda()
    computing_device = torch.device('cuda')
else:
    computing_device = torch.device('cuda')


def train():
    for epoch in range(epochs):
        ts = time.time()
        print(epoch)
        for iter, (X, tar, Y) in enumerate(train_loader):
            optimizer.zero_grad()

            # inputs = X.to(computing_device)
            inputs = X.cuda()
            labels = Y.cuda()
            # labels = Y.to(computing_device)

            print("Getting outputs")
            outputs = resnet_model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            #EARLY STOP TESTING CONDITION
            if iter > 5:
                break
            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        #torch.save(resnet_model, 'best_model')

        #val(epoch)
        resnet_model.train()


def val(epoch):
    print("Val")
    resnet_model.eval()
    # Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model


def test():
    None
    # Complete this function - Calculate accuracy and IoU
    # Make sure to include a softmax after the output from your model


if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()

# In[ ]:




