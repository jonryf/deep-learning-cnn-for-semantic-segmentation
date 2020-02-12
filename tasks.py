from runner import ModelRunner
from basic_fcn import FCN
from vgg11 import VGG
from unet import UNET
import torch
import torch.nn as nn
import sys


def task2(title=None):
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'model': FCN,
        'EPOCHS': 50,
        'batch_size': 4
    }
    print("Training FCN on", settings['EPOCHS'], "Epochs")
    runner = ModelRunner(settings)
    runner.load_data()
    runner.train()
    # runner.val()
    runner.plot(title=title)


def task3_1():
    settings_baseline = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': FCN,
        'EPOCHS': 10,
        'batch_size': 4,
        'LOAD_FROM_PATH': 'baseline_model.model'
    }

    settings_task3_1 = {
        'APPLY_TRANSFORMATIONS': True,
        'MODEL': FCN,
        'batch_size': 4,
        'EPOCHS': 10
    }

    baseline_runner = ModelRunner(settings_baseline)
    task_model = ModelRunner(settings_task3_1)

    # combine two plots
    task_model.plot(baseline_runner, names=["Baseline", "Model with transformations"])


def task_3_3():
    settings_baseline = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': FCN,
        'EPOCHS': 10,
        'batch_size': 4,
        'LOAD_FROM_PATH': 'baseline_model.model'
    }

    settings_task3_1 = {
        'APPLY_TRANSFORMATIONS': True,
        'MODEL': FCN,
        'batch_size': 4,
        'EPOCHS': 10
    }

    baseline_runner = ModelRunner(settings_baseline)
    task_model = ModelRunner(settings_task3_1)

    weights = [1]*30
    class_weights = torch.FloatTensor(weights).cuda()
    task_model.criterion = nn.CrossEntropyLoss(weight=class_weights)

    # combine two plots
    task_model.plot(baseline_runner, names=["Baseline", "Model with transformations"])

def task_3_4():
    settings_baseline = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': FCN,
        'EPOCHS': 10,
        'batch_size': 4,
        'LOAD_FROM_PATH': 'baseline_model.model'
    }
    settings_task3_4 = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': VGG,
        'batch_size': 4,
        'EPOCHS': 10
    }

    baseline_runner = ModelRunner(settings_baseline)
    task_model = ModelRunner(settings_task3_4)

    task_model.plot(baseline_runner, names=["Baseline", "VGG 11"])

def task_unet(title=None):
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'model': UNET,
        'EPOCHS': 50,
        'batch_size': 1,
        'imagesPerEpoch': 1
    }
    print("Training UNET on", settings['EPOCHS'], "Epochs")
    runner = ModelRunner(settings)
    runner.load_data()
    runner.train()
    # runner.val()
    runner.plot(title=title)

def test_task(title="TestRun"):
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'model': FCN,
        'EPOCHS': 8,
        'batch_size': 4,
        'imagesPerEpoch': 20
    }
    print("Training FCN on", settings['EPOCHS'], "Epochs")
    runner = ModelRunner(settings)
    runner.load_data()
    runner.train()
    # runner.val()
    runner.plot(title=title)

if __name__ == "__main__":
    test_task(title=sys.argv[1])