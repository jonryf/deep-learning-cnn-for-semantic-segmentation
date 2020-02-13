from dataloader import *
from utils import *
from dataloader import DataLoader
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

        self.criterion = loss.CrossEntropyLoss()
        self.model = settings['MODEL'](n_class=n_class)
        self.model.apply(init_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-3)

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
        val_dataset = CityScapesDataset('val.csv', self.transforms)
        test_dataset = CityScapesDataset('test.csv', self.transforms)

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

                lossSum += loss.data

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
            self.model.training_loss.append(lossSum)
            accuracy = accuracySum / totalImage
            if accuracy is None:
                accuracy = torch.tensor([0.0])
            self.model.training_acc.append(accuracy.item())
            print(totalImage*self.batch_size)
            print("-------------------------------------")
            print("Train epoch {}, time elapsed {}, loss {}, accuracy: {}".format(epoch, time.time() - ts, lossSum, accuracy.item()))
            print("Saving most recent model")

            torch.save(self.model, '{}lastEpochModel'.format(self.model_name))

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
                lossSum += self.criterion(outputs, labels).data.item()
                accuracySum += torch.sum(pixel_acc(outputs, labels))/self.batch_size
                torch.cuda.empty_cache()
            
        accuracy = accuracySum / totalImage
        if accuracy is None:
            accuracy = torch.tensor([0.0])
        print("Validation Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, lossSum, accuracy.item()))
        if self.bestValidationLoss is None or lossSum < self.bestValidationLoss:
            print("Saving best model")
            self.bestValidationLoss = lossSum
            torch.save(self.model, '{}bestModel'.format(self.model_name))
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

