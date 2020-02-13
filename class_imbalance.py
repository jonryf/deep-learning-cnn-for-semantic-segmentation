from basic_fcn import *
from dataloader import *
from utils import *
from dataloader import DataLoader
import torch.nn.modules.loss as loss
import time

train_dataset = CityScapesDataset('train.csv', None)
train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=1,
                                       num_workers=1,
                                       shuffle=True)

weights = np.zeros((34, 1))
for iter, (X, tar, Y) in enumerate(train_loader):
    if (iter % 100) == 0:
        print(iter)

    unique, counts = np.unique(Y.numpy(), return_counts=True)
    xx = 0
    for u, c in zip(unique, counts):
        weights[u] = (weights[u] + (c/2097152))

n_weights = weights
n_weights = 1 - n_weights / n_weights.sum()
n_weights = n_weights / n_weights.sum()


print(n_weights*34)
print((n_weights*34).tolist())
