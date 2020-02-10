from basic_fcn import *
from dataloader import *
from utils import *
from dataloader import DataLoader
import torch.nn.modules.loss as loss
import torch.optim as optim
import time


class ModelRunner:
    def __init__(self, settings):
        self.settings = settings
        print(settings)
        self.model_name = settings['MODEL'].__class__.__name__
        self.transforms = get_transformations() if settings['APPLY_TRANSFORMATIONS'] else None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.criterion = loss.CrossEntropyLoss()
        self.model = settings['MODEL'](n_class=n_class)
        self.model.apply(init_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-3)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.computing_device = torch.device('cuda')
        else:
            self.computing_device = torch.device('cpu')

        self.load_data()

    def load_data(self):
        train_dataset = CityScapesDataset('train.csv', self.transforms)
        val_dataset = CityScapesDataset('val.csv', self.transforms)
        test_dataset = CityScapesDataset('test.csv', self.transforms)

        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=1,
                                       num_workers=1,
                                       shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=1,
                                     num_workers=1,
                                     shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=1,
                                      num_workers=1,
                                      shuffle=True)


    def train(self):
        self.model.train()

        # log data to these variables
        self.model.training_loss = []
        self.model.validation_loss = []
        self.model.training_acc = []
        self.model.validation_acc = []

        for epoch in range(self.settings['EPOCHS']):
            ts = time.time()
            print(epoch)
            for iter, (X, tar, Y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                inputs = X.to(self.computing_device)

                labels = Y.to(self.computing_device)

                print("Getting outputs")
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                self.model.training_loss.append(loss)

                if iter > 20:
                    break
                if iter % 10 == 0:
                    print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

            print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
            print("Saving model")

            torch.save(self.model_name, self.model)

            self.val(epoch)

    def val(self, epoch):
        print("Val")
        self.model.eval()
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


if __name__ == "__main__":
    pass
    # val(0)  # show the accuracy before training
    # train()

# %%
