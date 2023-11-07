""" 
AlexNet for Binary Image Classification

This implementation adapts the renowned AlexNet architecture for binary image classification tasks.
The model is composed of a sequence of convolutional layers for feature extraction and fully connected layers for classification. 
It incorporates the binary cross-entropy loss function and is specifically tailored for binary classification scenarios.
The network employs the Adam optimizer for parameter updates
Input images are processed through the convolutional layers to extract meaningful features, and the model produces a single output, utilizing a sigmoid activation, representing the probability of an input belonging to the positive class.


"""



import torch
from torch import nn
import pytorch_lightning as pl

class AlexNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Restore the original configuration for 3-channel input
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # Increase the number of output units
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # Another fully connected layer
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # Yet another fully connected layer
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),  # Adjust the number of output units to 1
            nn.Sigmoid()  # Add Sigmoid activation to produce values in [0, 1]
        )
        
        # Binary cross-entropy loss function
        self.loss_func = nn.BCELoss()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the output from the convolutional layers
        x = self.classifier(x)
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1).float()  # Ensure y has the same shape as the model's output and is of data type float
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1).float()  # Ensure y has the same shape as the model's output and is of data type float
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log("test_loss", loss)
        return loss
