from runner import ModelRunner
from basic_fcn import FCN
from unet import UNET
import torch
import torch.nn as nn


def task2():
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'model': FCN,
        'EPOCHS': 100
    }
    print("Training FCN on", settings['EPOCHS'], "Epochs")
    runner = ModelRunner(settings)
    runner.load_data()
    runner.train()
    # runner.val()
    runner.plot()


def task3_1():
    settings_baseline = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': FCN,
        'EPOCHS': 10,
        'LOAD_FROM_PATH': 'baseline_model.model'
    }

    settings_task3_1 = {
        'APPLY_TRANSFORMATIONS': True,
        'MODEL': FCN,
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
        'LOAD_FROM_PATH': 'baseline_model.model'
    }

    settings_task3_1 = {
        'APPLY_TRANSFORMATIONS': True,
        'MODEL': FCN,
        'EPOCHS': 10
    }

    baseline_runner = ModelRunner(settings_baseline)
    task_model = ModelRunner(settings_task3_1)

    weights = [1]*30
    class_weights = torch.FloatTensor(weights).cuda()
    task_model.criterion = nn.CrossEntropyLoss(weight=class_weights)

    # combine two plots
    task_model.plot(baseline_runner, names=["Baseline", "Model with transformations"])

def task_unet():
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'model': UNET,
        'EPOCHS': 100
    }
    print("Training UNET on", settings['EPOCHS'], "Epochs")
    runner = ModelRunner(settings)
    runner.load_data()
    runner.train()
    # runner.val()
    runner.plot()


if __name__ == "__main__":
    task2()