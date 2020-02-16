from runner import ModelRunner
from basic_fcn import FCN
from vgg11 import VGG
from unet import UNET
from wnet import WNET
import torch
import torch.nn as nn
import sys
import io

def task2(title=None):
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': FCN,
        'EPOCHS': 50,
        'batch_size': 4,
        'learning_rate': 5e-3,
        'title': title
    }
    print("Training FCN on", settings['EPOCHS'], "Epochs")
    runner = ModelRunner(settings)
    runner.load_data()
    runner.train()
    # runner.val()


def task3_1(title="model with transformations"):
    settings_baseline = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': FCN,
        'EPOCHS': 50,
        'batch_size': 4,
        'LOAD_FROM_PATH': 'baseline_model.model',
        'learning_rate': 5e-3
    }

    settings_task3_1 = {
        'APPLY_TRANSFORMATIONS': True,
        'MODEL': FCN,
        'batch_size': 4,
        'EPOCHS': 50,
        'learning_rate': 5e-3,
        'title': title
    }

    #baseline_runner = ModelRunner(settings_baseline)
    runner = ModelRunner(settings_task3_1)
    runner.load_data()
    runner.train()
    #runner.val()

    # combine two plots
    #task_model.plot(baseline_runner, names=["Baseline", "Model with transformations"])


def task_3_3(title = 'task3-3'):
    settings_baseline = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': FCN,
        'EPOCHS': 10,
        'batch_size': 4,
        'LOAD_FROM_PATH': 'baseline_model.model',
        'learning_rate': 5e-3
    }

    settings_task3_1 = {
        'NAME': 'task 3-1',
        'APPLY_TRANSFORMATIONS': True,
        'MODEL': FCN,
        'batch_size': 4,
        'EPOCHS': 50,
        'learning_rate': 5e-3,
        'title': title
    }

    baseline_runner = ModelRunner(settings_baseline)
    runner = ModelRunner(settings_task3_1)

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
    runner.criterion = nn.CrossEntropyLoss(weight=class_weights)
    runner.train()
    # runner.val()

    # combine two plots
    #task_model.plot(baseline_runner, names=["Baseline", "Model with transformations"])


def task_3_4(title):
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': VGG,
        'EPOCHS': 50,
        'batch_size': 4,
        'learning_rate': 5e-4,
        'title': title
    }
    print("Training VGG on", settings['EPOCHS'], "Epochs")
    runner = ModelRunner(settings)
    runner.load_data()
    runner.train()
    # runner.val()

def task_unet(title='Unet'):
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': UNET,
        'EPOCHS': 50,
        'batch_size': 1,
        'learning_rate': 5e-5,
        'title': title
    }
    print("Training UNET on", settings['EPOCHS'], "Epochs")
    runner = ModelRunner(settings)
    runner.load_data()
    runner.train()
    # runner.val()

def task_wnet(title='Wnet'):
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': WNET,
        'EPOCHS': 100,
        'batch_size': 1,
        'learning_rate': 5e-4,
        'title': title
    }
    print("Training WNET on", settings['EPOCHS'], "Epochs")
    runner = ModelRunner(settings)
    runner.load_data()
    runner.train()
    # runner.val()

def test_task(title="TestRun"):
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': FCN,
        'EPOCHS': 10,
        'batch_size': 1,
        'imagesPerEpoch': 10,
        'learning_rate': 5e-3,
        'title': title

    }
    print("Training FCN on", settings['EPOCHS'], "Epochs")
    runner = ModelRunner(settings)
    runner.load_data()
    runner.train()
    # runner.val()

def test_vgg(title="VGG_Test"):
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': VGG,
        'EPOCHS': 10,
        'batch_size': 1,
        'imagesPerEpoch': 10,
        'learning_rate': 5e-3,
        'title': title
    }
    print("Training VGG on", settings['EPOCHS'], "Epochs")
    runner = ModelRunner(settings)
    runner.load_data()
    runner.train()
    # runner.val()

def continue_training(title, fileName, epochs, batchSize, learning_rate):

    model = torch.load('./{}'.format(fileName))

    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': model,
        'EPOCHS': epochs,
        'batch_size': batchSize,
        'learning_rate': learning_rate,
        'loaded': True,
        'title': title
    }

    runner = ModelRunner(settings)
    runner.load_data()
    runner.train()

def test_acc(title, fileName, batchSize):

    model = torch.load('./{}'.format(fileName))

    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': model,
        'EPOCHS': 1,
        'batch_size': batchSize,
        'learning_rate': 5e-3,
        'title': title,
        'loaded': True
    }
    print("Getting Accuracy for {}".format(fileName))
    runner = ModelRunner(settings)
    runner.load_data()
    runner.test()
    # runner.val()
    

if __name__ == "__main__":
    task = input("Which task? (2, 3.1, 3.2, 3.3, 3.4, 3.5, 4: continue_training), 5: get pixel accuracy: ")
    title = input("Name of the graph:")
    if task == '2':
        task2(title)
    elif task == '3.1':
        task3_1(title)
    elif task == '3.2':
        task_wnet(title)
    elif task == '3.3':
        task_3_3(title)
    elif task == '3.4':
        task_3_4(title)
        #test_vgg(title)
    elif task == '3.5':
        task_unet(title)
    elif task == 'test_task':
        test_task(title)
    elif task == '4':
        fileName = input("Name of model file to load (don't include './''): ")
        epochs = int(input("Enter number of epochs: "))
        batchSize = int(input("Please enter a batch size (1 for Unet, 4 for FCN): "))
        learning_rate = float(input("Please enter a learning rate (5e-3): "))
        continue_training(title, fileName, epochs, batchSize, learning_rate)
    elif task == '5':
        fileName = input("Name of model file to load (don't include './''): ")
        batchSize = int(input("Enter a batch size: "))
        test_acc(title, fileName, batchSize)


    print("Thank you, exiting program")
