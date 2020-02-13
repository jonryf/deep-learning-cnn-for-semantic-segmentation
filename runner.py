from dataloader import *
from utils import *
from dataloader import DataLoader
from vgg11 import VGG
import torch.nn.modules.loss as loss
import torch.optim as optim
import time


class ModelRunner:
    def __init__(self, settings):
        self.settings = settings
        self.model_name = settings['MODEL'].__name__
        self.transforms = get_transformations() if settings['APPLY_TRANSFORMATIONS'] else None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.bestValidationLoss = None
        self.batch_size = settings['batch_size']
        self.learning_rate = settings['learning_rate']
        self.title = settings['title']

        self.criterion = loss.CrossEntropyLoss()
        self.model = settings['MODEL'](n_class=n_class)

        # account for VGG needing different init_weights
        transfer = (settings['MODEL'] == VGG)
        if transfer:
            self.model.apply(init_weights_transfer)
        else:
            self.model.apply(init_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.model = self.model.cuda()
            self.computing_device = torch.device('cuda')
        else:
            self.computing_device = torch.device('cpu')

        print(self.model_name)
        self.load_data()

    def load_data(self):
        train_dataset = CityScapesDataset('train.csv', self.transforms)
        val_dataset = CityScapesDataset('val.csv', None)
        test_dataset = CityScapesDataset('test.csv', None)

        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=self.batch_size,
                                       num_workers=4,
                                       shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=self.batch_size,
                                     num_workers=4,
                                     shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=4,
                                      shuffle=True)


    def train(self):
        self.model.train()

        # log data to these variables
        self.model.training_loss = []
        self.model.training_acc = []
        self.model.validation_acc = []
        self.model.validation_loss = []

        for epoch in range(self.settings['EPOCHS']):
            self.model.train()
            ts = time.time()
            lossSum = 0
            accuracySum = 0
            totalImage = 0
            for iter, (X, tar, Y) in enumerate(self.train_loader):

                
                self.optimizer.zero_grad()

                if('imagesPerEpoch' in self.settings):
                    if iter*self.batch_size > self.settings['imagesPerEpoch']:
                        break
                

                #inputs = X.to(computing_device)
                inputs = X.cuda()
                labels = Y.cuda()
                #labels = Y.to(computing_device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                lossSum += loss.item()

                accuracies = pixel_acc(outputs, labels)

                accuracySum += torch.sum(accuracies)/self.batch_size

                torch.cuda.empty_cache()
                
                loss.backward()
                self.optimizer.step()

                

                totalImage += 1

                if iter % 100 == 0:
                    None
                    print("Iter", iter, "Done")
                    #print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
            lossSum = lossSum / totalImage

            self.model.training_loss.append(lossSum)
            accuracy = accuracySum / totalImage # totalImage?
            if accuracy is None:
                accuracy = torch.tensor([0.0])
            self.model.training_acc.append(accuracy.item())
            print(totalImage*self.batch_size)
            print("-------------------------------------")
            print("Train epoch {}, time elapsed {}, loss {}, accuracy: {}".format(epoch, time.time() - ts, lossSum, accuracy.item()))
            print("Saving most recent model")

            torch.save(self.model, '{}lastEpochModel'.format(self.title))

            self.val(epoch)

    def val(self, epoch):
        self.model.eval()
        vals = self.val_loader
        lossSum = 0
        accuracySum = 0
        totalImage=0
        for iter, (X, tar, Y) in enumerate(vals):
            if 'imagesPerEpoch' in self.settings:
                if iter * self.batch_size > self.settings['imagesPerEpoch']:
                    break
            with torch.no_grad():
                inputs = X.cuda()
                labels = Y.cuda()
    #             print(torch.cuda.memory_allocated(device=None))
                totalImage += 1
                outputs = self.model(inputs)
                lossSum += self.criterion(outputs, labels).item()
                accuracySum += torch.sum(pixel_acc(outputs, labels))/self.batch_size
                torch.cuda.empty_cache()

        lossSum = lossSum / totalImage
        accuracy = accuracySum / totalImage
        if accuracy is None:
            accuracy = torch.tensor([0.0])
        print("Validation Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, lossSum, accuracy.item()))
        if self.bestValidationLoss is None or lossSum < self.bestValidationLoss:
            print("Saving best model")
            self.bestValidationLoss = lossSum
            torch.save(self.model, '{}{}bestModel'.format(self.settings.get('NAME', ''), self.title))
        self.model.validation_loss.append(lossSum)
        self.model.validation_acc.append(accuracy.item())
        # Complete this function - Calculate loss, accuracy and IoU for every epoch
        # Make sure to include a softmax after the output from your model

    def test(self):
        None
        # Complete this function - Calculate accuracy and IoU
        # Make sure to include a softmax after the output from your model

    def plot(self, compare_to=None, names=None, title=None):
        if compare_to is None:
            plot(self.model, title=title)
        else:
            multi_plots([self.model, compare_to.model], names)

