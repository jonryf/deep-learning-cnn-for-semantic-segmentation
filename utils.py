from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def getClassFromChannels(preds):
    return torch.argmax(preds, axis=1)

'''
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        # Complete this function
        intersection = # intersection calculation
        union = #Union calculation
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            # Append the calculated IoU to the list ious
    return ious
'''

# pred - pred(images, height, width) = prediction class
# target - target(images, height, width) = target class
# returns - (images, percentCorrect)
def pixel_acc(pred, target):
    diff = pred - target
    correct = torch.where(diff == 0, 1, 0)
    s = torch.sum(correct, axis=1)
    s = torch.sum(s, axis=1)
    return s.type(torch.DoubleTensor)/ float(pred.size()[1] * pred.size()[2])


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
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])


def graph_error(means, stds, labels, legends, title, show=True):
    """
    Create a plot with error bar
    @param means: mean data
    @param stds: standard deviation
    @param labels: labels the axises
    @param legends: legends for the plot
    @param title: title of the plot
    @param show: if the plot should be displayed right away. Set to false to display multiple graphs in same plot
    """
    for i in range(2):
        plt.errorbar(np.arange(1, len(means[i])+1), means[i], yerr=stds[i])
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legends)
    if show:
        plt.show()


def graph_plot(data, labels, legends, title, show=True):
    """
    Plot multiple graphs in same plot

    @param data: data of the graphs to be plotted
    @param labels: x- and y-label
    @param legends: legends for the graphs
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


def plot_loss(model, title, show=True):
    """
    Plot loss and accuracy graphs

    @param model: trained model to plot
    """
    # plot the loss
    graph_plot([model.training_loss, model.validation_loss],
               ["Epoch", "Cross-entropy loss"], ["Training loss", "Validation loss"], title, show)


def plot_acc(model, title, show=True):
    # plot the accuracy
    graph_plot([model.training_acc, model.validation_acc],
               ["Epoch", "Accuracy"], ["Training accuracy", "Validation accuracy"], title, show)


def plot(model, title=""):
    plot_loss(model, title, show=True)
    #plot_acc(model, title, show=True)


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