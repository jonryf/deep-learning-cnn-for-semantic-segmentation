from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataloader import labels_classes
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from dataloader import labels_classes

#size (images, features, height, width)
def getOneHotPredictionsFromProbabilites(preds):
    ret = torch.zeros(preds.size()).cuda()
    maxPreds = torch.argmax(preds, axis=1) # dimmensinos (images, height, width) = feature prediction
    for imageIdx in range(0,maxPreds.size()[0]):
        for heightIdx in range(0,maxPreds.size()[1]):
            for widthIdx in range(0,maxPreds.size()[2]):
                ret[imageIdx, maxPreds[imageIdx, heightIdx, widthIdx], heightIdx, widthIdx] = 1
    return ret




def getClassFromChannels(preds):
    return torch.argmax(preds, axis=1)

# pred is predicted probabilites
# target is one hot encoding of 
def iou(pred, target):
    # i didt like the way they did it, looks like it was only for one example
    # ious = []
    # for cls in range(n_class):
    #     # Complete this function
    #     intersection = # intersection calculation
    #     union = #Union calculation
    #     if union == 0:
    #         ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
    #     else:
    #         # Append the calculated IoU to the list ious
    # return ious
    oneHotPReds = getOneHotPredictionsFromProbabilites(pred)


    # numerator
    classTp = oneHotPReds * target # get intersection
    classTp = torch.sum(classTp, axis=2)
    classTp = torch.sum(classTp, axis=2) # sum accross image height and width


    # denominator
    union = oneHotPReds + target # get union
    union[torch.where(union > 1)] = 1 # account for intersection points
    classUnion = torch.sum(union, axis=2)
    classUnion = torch.sum(classUnion, axis=2) # sum across height and width of image

    return classTp.float() / classUnion.float()


# pred - pred(images, classes, height, width) = prediction class
# target - target(images, classes, height, width) = target class
# returns - (images) = pecent
def pixel_acc(pred, target):

    a = getClassFromChannels(pred) #[target != 255]
    b = target #[target != 255] #getClassFromChannels(target)

    diff = a - b

    if torch.cuda.is_available():
        x = torch.tensor(1).cuda()
        y = torch.tensor(0).cuda()
    else:
        x = torch.tensor(1)
        y = torch.tensor(0)
    
    correct = torch.where(diff == 0, x, y)



    s = torch.sum(correct, axis=1)
    s = torch.sum(s, axis=1)

    return s.float() / float(a.size()[1] * a.size()[2])

def exclusion_pixel_acc(pred, target):
    predClass = getClassFromChannels(pred)
    correct = 0
    total = 0

    for c in range(0, pred.size()[1]):
        if not labels_classes[c].ignoreInEval:
            correct += ((predClass == target) & (target == c)).sum().item()
            total += (target == c).sum().item()

    return correct / total

def better_IoU(pred, target):
    predClass = getClassFromChannels(pred)
    IoU = torch.tensor([])

    for c in range(0, pred.size()[1]):
        if not labels_classes[c].ignoreInEval:
            GT = (target == c)
            TP = ((predClass == target) & GT).sum().item()
            FP = ((predClass == c) & (predClass != target)).sum().item()
            FN = (GT & (predClass != target)).sum().item()
            #IoU.append(TP / (TP + FP + FN))
            denom = TP + FP + FN
            num = TP
            val = 0
            if  denom == 0:
                val = 0
            else:
                val = num / denom
            IoU = torch.cat((IoU, torch.tensor([float(val)])))
    return IoU




def init_weights(model):
    """
    Apply weights to a Pytorch model

    :param model: Pytorch model
    """

    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(model.weight.data)
        nn.init.zeros_(model.bias.data)


def init_weights_transfer(model):
    """
    Apply weights to a Pytorch model

    :param model: Pytorch model
    """
    if isinstance(model, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(model.weight.data)
        nn.init.zeros_(model.bias.data)



def graph_plot(data, labels, legends, time, title="", show=True):
    """
    Plot multiple graphs in same plot

    @param data: data of the graphs to be plotted
    @param labels: x- and y-label
    @param legends: legends for the graphs
    :param show:
    :param title:
    """
    x = np.arange(1, len(data[0]) + 1)
    for to_plot in data:
        plt.plot(x, to_plot)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legends)
    if show:
        plt.show()
        plt.savefig('{} -{}.png'.format(time, title))


def plot_loss(model, title, time, show=True):
    """
    Plot loss and accuracy graphs

    @param model: trained model to plot
    """
    # plot the loss
    plt.clf()
    graph_plot([model.training_loss, model.validation_loss],
               ["Epoch", "Cross-entropy loss"], ["Training loss", "Validation loss"], time,"Loss for " + title, show)



def plot_acc(model, title, time, show=True):
    # plot the accuracy
    plt.clf()
    graph_plot([model.training_acc, model.validation_acc],
               ["Epoch", "Accuracy"], ["Training accuracy", "Validation accuracy"], time, "Accuracy for " + title , show)


def plot(model, time, title="",):
    plot_loss(model, title, time, show=True)
    plot_acc(model, title, time, show=True)


def multi_plots(models, names):
    """
    Plot loss and accuracy graphs

    @param model: trained model to plot
    """
    # plot the loss
    losses = []
    loss_labels = []

    acc = []

    for model, name in zip(models, names):
        losses += model.training_loss
        losses += model.validation_loss
        loss_labels += "Training loss - " + name
        loss_labels += "validation loss - " + name

        acc += model.training_acc
        acc += model.validation_acc

    graph_plot(losses,
               ["Epoch", "Cross-entropy loss"], loss_labels)

    # plot the accuracy
    graph_plot([model.training_acc, model.validation_acc],
               ["Epoch", "Accuracy"], ["Training accuracy", "Validation accuracy"])


def visualize(image_idx, segmentation, csv='test.csv'):
    image = pd.read_csv(csv).iloc[image_idx, 0]
    image = Image.open(image).convert('RGB')
    segmentation = segmentation.numpy()[0]
    plt.figure(figsize = (20,40))
    plt.imshow(np.asarray(image), interpolation='none')
    plt.imshow(np.asarray(segmentation), interpolation='none', alpha=0.7)
    plt.axis('off')
    plt.savefig('visualization.png')