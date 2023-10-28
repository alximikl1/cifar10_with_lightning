import wandb
wandb.login()

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from model.Model import CNNNet, CIFAR10Model
from data_model.DataModule import CIFAR10DataModule

EPOCHS = 12

dm = CIFAR10DataModule()
net = CIFAR10Model(CNNNet(), EPOCHS)

wandb_logger = WandbLogger(project="CIFAR10_")

early_stopping1 = EarlyStopping('val_loss')
early_stopping2 = EarlyStopping('val_acc', patience=2, mode="max")
cb = LearningRateMonitor(logging_interval='step')

trainer = L.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",
    devices=1,
    logger=wandb_logger,
    callbacks=[early_stopping1, early_stopping2, cb]
)

trainer.fit(net, dm)

trainer.test(net, dm)

wandb.finish()