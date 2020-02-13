from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

#size (images, features, height, width)
def getOneHotPredictionsFromProbabilites(preds):
    ret = torch.zeros(preds.size())
    maxPreds = torch.argmax(preds, axis=1) # dimmensinos (images, height, width) = feature prediction
    for imageIdx in maxPreds.size()[0]:
        for heightIdx in maxPreds.size()[1]:
            for widthIdx in maxPreds.size()[2]:
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
    classPreds = getClassFromChannels(pred) #[target != 255]
    classTarget = target #[target != 255] #getClassFromChannels(target)
    diff = classPreds - classTarget
    x = torch.tensor(1).cuda()
    y = torch.tensor(0).cuda()
    correct = torch.where(diff == 0, x, y)
    s = torch.sum(correct, axis=1)
    s = torch.sum(s, axis=1)
    return s.float() / float(pred.size()[2] * pred.size()[3])


def init_weights(model):
    """
    Apply weights to a Pytorch model

    :param model: Pytorch model
    """
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(model.weight.data)
        nn.init.zeros_(model.bias.data)


def get_transformations():
    """
    Compose a set of transformations

    :return: transformations
    """
    return transforms.Compose([
            transforms.ColorJitter(),
            transforms.RandomCrop((512, 1024)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip()])


def graph_plot(data, labels, legends, title="", show=True):
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
        plt.savefig('{} -{}.png'.format(datetime.now(), title))


def plot_loss(model, title, show=True):
    """
    Plot loss and accuracy graphs

    @param model: trained model to plot
    """
    # plot the loss
    plt.clf()
    graph_plot([model.training_loss, model.validation_loss],
               ["Epoch", "Cross-entropy loss"], ["Training loss", "Validation loss"], "Loss for " + title, show)



def plot_acc(model, title, show=True):
    # plot the accuracy
    plt.clf()
    graph_plot([model.training_acc, model.validation_acc],
               ["Epoch", "Accuracy"], ["Training accuracy", "Validation accuracy"], "Accuracy for " + title , show)


def plot(model, title=""):
    plot_loss(model, title, show=True)
    plot_acc(model, title, show=True)


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