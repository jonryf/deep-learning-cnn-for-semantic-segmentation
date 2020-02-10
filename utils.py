from torchvision import transforms
import torch
import torch.nn as nn

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


def pixel_acc(pred, target):
    #Complete this function
'''


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