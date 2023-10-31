from torch import argmax
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import lightning as L

from model import EPOCHS

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()

        # Convolutional layers
                            #Init_channels, channels, kernel_size, padding)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Pooling layers
        self.pool = nn.MaxPool2d(2,2)

        # FC layers
        # Linear layer (64x4x4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)

        # Linear Layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)

        # Dropout layer
        self.dropout = nn.Dropout(0.25)
        
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))

        # Flatten the image
        x = x.view(-1, 64*4*4)
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


class CIFAR10Model(L.LightningModule):
    def __init__(self, model, learning_rate=0.0015):
        super().__init__()

        # We take in input dimensions as parameters and use those to dynamically build model.
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        data, target = batch

        output = self.model(data)
        loss = self.criterion(output, target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch

        output = self.model(data)
        loss = self.criterion(output, target)

        preds = argmax(output, dim=-1)
        acc = accuracy(preds, target, task="multiclass", num_classes=10)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        data, target = batch
        
        output = self.model(data)
        loss = self.criterion(output, target)

        preds = argmax(output, dim=-1)
        acc = accuracy(preds, target, task="multiclass", num_classes=10)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.012, epochs=1, steps_per_epoch=EPOCHS)
        return [optimizer], [scheduler]