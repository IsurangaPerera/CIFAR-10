import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import SWALR
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Network import ResUnitBlock, ResNetV2Prop
from torchvision import transforms
from PIL import Image


class MyModel(object):

    def __init__(self, configs):
        self.model_configs = configs
        self.network = torch.nn.DataParallel(ResNetV2Prop(ResUnitBlock))
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=configs['initial_lr'], momentum=0.9,
                                         weight_decay=5e-4)

    @staticmethod
    def score(outputs, target):
        _, outputs = outputs.max(1)
        return torch.eq(outputs, target).sum().item()

    @staticmethod
    def checkpoint_model(network_state, checkpoint_dir, epoch):
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        state = {
            'net': network_state,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_dir + 'epoch' + str(epoch) + '_ckpt.pth')

    def update_lr(self, epoch, initial_learning_rate):
        lr = initial_learning_rate / ((epoch // 50) + 1)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, train, configs):
        self.network = self.network.to('cuda')
        torch.backends.cudnn.benchmark = True

        batch_size = configs['batch_size']
        swa_start = 225
        swa_scheduler = SWALR(self.optimizer, swa_lr=0.005)
        train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)

        scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        criterion = nn.CrossEntropyLoss()

        for epoch in range(configs['max_epoch']):
            self.network.train()
            with tqdm(total=len(train_loader)) as pbar:
                for _, (x_train, y_train) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    y_preds = self.network(x_train.to('cuda'))
                    loss = criterion(y_preds, y_train.to('cuda'))
                    loss.backward()
                    self.optimizer.step()
                    pbar.set_description('[Epoch %d/%d] Loss %.2f' % (epoch, configs['max_epoch'], loss))
                    pbar.update(1)

            self.update_lr(epoch, configs['initial_lr'])
            self.checkpoint_model(self.network.state_dict(), self.model_configs['save_dir'], epoch)

    def evaluate(self, data):
        self.network.eval()
        loader = torch.utils.data.DataLoader(data, 128, shuffle=False)
        total, correct = 0, 0
        with torch.no_grad():
            accuracy = 0
            for idx, (x, y) in enumerate(loader):
                outputs = self.network(x.to('cuda'))
                correct += self.score(outputs, y.to('cuda'))
                total += len(y)
                accuracy = correct / total
        return accuracy, correct, total

    def predict_prob(self, x):
        norm_mean = (0.4914, 0.4822, 0.4465)
        norm_std = (0.2470, 0.2435, 0.2616)
        self.network.eval()
        outputs = torch.empty(len(x), 10)
        with torch.no_grad():
            for idx in tqdm(range(x.shape[0])):
                x_train = x[idx].reshape((3, 32, 32))
                image = np.transpose(x_train, [1, 2, 0])

                current_transform = transforms.Compose([  # defualt transform which only normalizes the data set
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std)  # normalize dataset
                ])

                image = Image.fromarray(image)
                image = current_transform(image)
                image = np.transpose(image, [2, 0, 1])

                image = torch.tensor(image).type(torch.FloatTensor).unsqueeze(0).to('cuda')
                output = self.network(image.float().cuda())
                outputs[idx * len(output):(idx + 1) * len(output)] = output

        return torch.nn.functional.softmax(outputs, dim=1).to('cpu').numpy()
