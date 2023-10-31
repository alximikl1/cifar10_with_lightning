import lightning as L
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10

from model import BATCH_SIZE

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.num_classes = 10

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True, transform=self.transform)
        CIFAR10(self.data_dir, train=False, download=True, transform=self.transform)

    def setup(self, stage):
        if stage == "fit":
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(cifar10_full, [0.8, 0.2])

        if stage == "test":
            self.mnist_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE, num_workers=1)