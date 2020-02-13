from runner import ModelRunner
from basic_fcn import FCN
from unet import UNET
import torch
import torch.nn as nn
import sys


def task2(title=None):
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': FCN,
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
        'imagesPerEpoch': 20,
        'LOAD_FROM_PATH': 'baseline_model.model'
    }

    settings_task3_1 = {
        'APPLY_TRANSFORMATIONS': True,
        'MODEL': FCN,
        'batch_size': 2,
        'EPOCHS': 10,
        'imagesPerEpoch': 20

    }

    # baseline_runner = ModelRunner(settings_baseline)
    runner = ModelRunner(settings_task3_1)
    runner.load_data()
    runner.train()
    # runner.val()
    runner.plot(title="Loss, model with transformers")

    # combine two plots
    # task_model.plot(baseline_runner, names=["Baseline", "Model with transformations"])


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

    weights = [[1.030186615981065], [0.9830729949422731], [1.016867434679172], [1.0147616646506572],
               [1.016472345162264], [1.027360477860356], [1.0178136572899759], [0.6940124437406467],
               [0.9748015312095744], [1.0238518768574771], [1.0284470043760359], [0.8221235861509911],
               [1.0243251027689355], [1.022301386383189], [1.030212666234929], [1.0273536290750878],
               [1.0297476987302048], [1.0191083790832787], [1.0302204822358632], [1.028407722241951],
               [1.025274864130722], [0.8850169015859631], [1.0197425056226328], [0.9936456645618787],
               [1.0191848517083504], [1.0290705755985148], [0.9665053915048576], [1.0278635515485495],
               [1.0281578344700142], [1.029891278320577], [1.0300882554157473], [1.0281786972310123],
               [1.02940316600717], [1.0265277626400904]]
    class_weights = torch.FloatTensor(weights).cuda()
    task_model.criterion = nn.CrossEntropyLoss(weight=class_weights)

    # combine two plots
    task_model.plot(baseline_runner, names=["Baseline", "Model with transformations"])


def task_unet(title=None):
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': UNET,
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
        'MODEL': FCN,
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
    # task2(title=sys.argv[1])
    task3_1()
