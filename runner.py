from dataloader import *
from utils import *
from dataloader import DataLoader
from vgg11 import VGG
import torch.nn.modules.loss as loss
import torch.optim as optim
import time
from datetime import datetime


class ModelRunner:
    def __init__(self, settings):
        self.settings = settings
        self.transforms = settings['APPLY_TRANSFORMATIONS'] 
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.bestValidationLoss = None
        self.batch_size = settings['batch_size']
        self.learning_rate = settings['learning_rate']
        self.title = settings['title']

        if 'loaded' in settings:
            if not settings['loaded']:
                self.model = settings['MODEL'](n_class=n_class)
                self.model_name = settings['MODEL'].__name__
            else:
                self.model = settings['MODEL']
        else:
            self.model = settings['MODEL'](n_class=n_class)
            self.model_name = settings['MODEL'].__name__

            

        self.start_time = datetime.now()



        self.criterion = loss.CrossEntropyLoss()
        

        # account for VGG needing different init_weights
        transfer = (settings['title'] == 'VGG')
        if transfer:
            if 'loaded' in settings:
                if not settings['loaded']:
                    self.model.apply(init_weights_transfer)
            else:
                self.model.apply(init_weights_transfer)
        else:
            if 'loaded' in settings:
                if not settings['loaded']:
                    self.model.apply(init_weights)
            else:
                self.model.apply(init_weights)



        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.model = self.model.cuda()
            self.computing_device = torch.device('cuda')
        else:
            self.computing_device = torch.device('cpu')

        if 'loaded' in settings:
            if not settings['loaded']:
                print(self.model_name)
            else:
                print("Loaded Model")
        else:
            print(self.model_name)

        self.load_data()

    def load_data(self):
        train_dataset = CityScapesDataset('train.csv', self.transforms)
        val_dataset = CityScapesDataset('val.csv', None)
        test_dataset = CityScapesDataset('test.csv', None)

        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=self.batch_size,
                                       num_workers=3,
                                       shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=self.batch_size,
                                     num_workers=3,
                                     shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=1,
                                      num_workers=3,
                                      shuffle=False)


    def train(self):
        self.model.train()

        # log data to these variables
        if 'loaded' in self.settings:
            if not self.settings['loaded']:
                self.model.training_loss = []
                self.model.training_acc = []
                self.model.validation_acc = []
                self.model.validation_loss = []
        else:
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

                if('imagesPerEpoch' in self.settings):
                    if iter*self.batch_size > self.settings['imagesPerEpoch']:
                        break

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
  
        self.model.validation_loss.append(lossSum)
        self.model.validation_acc.append(accuracy.item())

        if self.bestValidationLoss is None or lossSum < self.bestValidationLoss:
            print("Saving best model")
            self.bestValidationLoss = lossSum
            torch.save(self.model, '{}{} - Best Model for {}'.format(self.start_time, self.settings.get('NAME', ''), self.title))


        torch.save(self.model, '{} - Last Epoch Model for {}'.format(self.start_time,self.title))
        print("Saving most recent model")

        self.plot(title=self.title)
        # Complete this function - Calculate loss, accuracy and IoU for every epoch
        # Make sure to include a softmax after the output from your model

    def test(self):
        self.model.eval()
        vals = self.test_loader
        accuracySum = 0
        totalImages = 0
        for iter, (X, tar, Y) in enumerate(vals):
            print(iter)
            with torch.no_grad():
                inputs = X.cuda()
                labels = Y.cuda()
    #             print(torch.cuda.memory_allocated(device=None))
                totalImages += 1
                outputs = self.model(inputs)
                accuracySum += exclusion_pixel_acc(outputs, labels)
                torch.cuda.empty_cache()
        accuracy = accuracySum / totalImages
        if accuracy is None:
            accuracy = torch.tensor([0.0])
        print("Validation Accuracy: {}".format(accuracy))
        # Complete this function - Calculate accuracy and IoU
        # Make sure to include a softmax after the output from your model

    def test_iou(self):
        self.model.eval()
        vals = self.val_loader
        IoU = 0
        totalImages = 0
        for iter, (X, tar, Y) in enumerate(vals):
            if 'imagesPerEpoch' in self.settings:
                if iter * self.batch_size > self.settings['imagesPerEpoch']:
                    break
            with torch.no_grad():
                if('imagesPerEpoch' in self.settings):
                    if iter*self.batch_size > self.settings['imagesPerEpoch']:
                        break

                inputs = X.cuda()
                labels = Y.cuda()
                #targets = tar.cuda()

                totalImages += 1

                outputs = self.model(inputs)

                IoU += better_IoU(outputs, labels)

                torch.cuda.empty_cache()
        IoU = IoU / totalImages
        c = 0
        for i in range(0, len(labels_classes)):
            if not labels_classes[i].ignoreInEval:
                print("{} Acc: {}".format(labels_classes[i].name, IoU[c]))
                c+=1
        print(torch.sum(IoU, dim=0)/IoU.size()[0])
    
    def visualization(self):
        self.model.eval()
        test = self.test_loader
        for iter, (X, tar, Y) in enumerate(test):
            with torch.no_grad():
                '''
                inputs = X.cuda()

                greyscale = torch.mean(inputs, dim=1).cpu()
                greyscale = greyscale[0,:,:]

                plt.figure()
                plt.imshow(greyscale, cmap="binary")

                output = self.model(inputs)

                outputClasses = getClassFromChannels(output).cpu()
                outputClasses = outputClasses[0, :, :]
                plt.imshow(outputClasses, alpha = 0.5, cmap='Set1')
                plt.colorbar()
                plt.savefig('%s Visualization' %self.title)
                '''
                
                inputs = X.cuda()

                greyscale = inputs.cpu()
                greyscale = greyscale[0,:,:]

                greyscale = transforms.ToPILImage()(greyscale)

                output = self.model(inputs)
                outputClasses = getClassFromChannels(output).cpu()
                outputClasses = outputClasses[0, :, :]


                seg = torch.zeros([4,inputs.size()[2], inputs.size()[3]])

                for i in range(0, inputs.size()[2]):
                    for j in range(0, inputs.size()[3]):
                        seg[0][i][j] = labels_classes[outputClasses[i][j]].color[0]
                        seg[1][i][j] = labels_classes[outputClasses[i][j]].color[1]
                        seg[2][i][j] = labels_classes[outputClasses[i][j]].color[2]
                        seg[3][i][j] = 0.5
                segImage = transforms.ToPILImage()(seg)

                greyscale.paste(segImage, (0,0), segImage)
                greyscale.save("Vis.png")



            if iter > 0:
                break

    def plot(self, compare_to=None, names=None, title=None):
        if compare_to is None:
            plot(self.model, title=title, time=self.start_time)
        else:
            multi_plots([self.model, compare_to.model], names)

