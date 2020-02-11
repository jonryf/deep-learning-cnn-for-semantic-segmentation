from dataloader import *
from utils import *
from dataloader import DataLoader
import torch.nn.modules.loss as loss
import torch.optim as optim
import time


class ModelRunner:
    def __init__(self, settings):
        self.settings = settings
        self.model_name = settings['model'].__name__
        self.transforms = get_transformations() if settings['APPLY_TRANSFORMATIONS'] else None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.criterion = loss.CrossEntropyLoss()
        self.model = settings['model'](n_class=n_class)
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
                                       batch_size=4,
                                       num_workers=4,
                                       shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=4,
                                     num_workers=4,
                                     shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=4,
                                      num_workers=4,
                                      shuffle=True)


    def train(self):
        self.model.train()

        # log data to these variables
        self.model.training_loss = []
        self.model.validation_loss = []

        for epoch in range(self.settings['EPOCHS']):
            ts = time.time()
            lossSum = 0
            for iter, (X, tar, Y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                #inputs = X.to(computing_device)
                inputs = X.cuda()
                labels = Y.cuda()
                #labels = Y.to(computing_device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                lossSum += loss.data
                loss.backward()
                self.optimizer.step()


                # if iter > 1:
                #     break
                if iter % 100 == 0:
                    None
                    print("Iter", iter, "Done")
                    #print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
            self.model.training_loss.append(lossSum)
            print("-------------------------------------")
            print("Train epoch {}, time elapsed {}, loss {}".format(epoch, time.time() - ts, lossSum))
            print("Saving model")

            torch.save(self.model, self.model_name)

            self.val(epoch)

    def val(self, epoch):
        self.model.eval()
        vals = self.val_loader
        lossSum = 0
        for iter, (X, tar, Y) in enumerate(vals):
            with torch.no_grad():
                inputs = X.cuda()
                labels = Y.cuda()
    #             print(torch.cuda.memory_allocated(device=None))
                outputs = self.model(inputs)
                lossSum += self.criterion(outputs, labels).data.item()
            # if iter > 1:
            #     break
        print("Validation Epoch:", epoch, ", Loss: ", lossSum)
        self.model.validation_loss.append(lossSum)
        # Complete this function - Calculate loss, accuracy and IoU for every epoch
        # Make sure to include a softmax after the output from your model

    def test(self):
        None
        # Complete this function - Calculate accuracy and IoU
        # Make sure to include a softmax after the output from your model

    def plot(self, compare_to=None, names=None):
        if compare_to is None:
            plot(self.model)
        else:
            multi_plots([self.model, compare_to.model], names)

