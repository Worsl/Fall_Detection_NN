"""

 implements a simple feedforward neural network (FNN). 
 It consists of an input layer, a hidden layer with ReLU activation, 
 and an output layer with a sigmoid activation. 
 
 
 This architecture is suitable for binary classification tasks, and it takes a flattened input image 
 and produces a single output representing the probability of belonging to the positive class.


"""

import wandb
import torchvision
import torch
import torchmetrics
from torch import nn
import pytorch_lightning as pl

class SimpleFallDetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.input_size = (3, 224, 224)  # Input size based on image dimensions (3 channels, 224x224 pixels)

        # Define a feedforward neural network
        self.fc1 = nn.Linear(self.input_size[0] * self.input_size[1] * self.input_size[2], 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  # 64 hidden units to 1 output (binary classification)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation

        # Binary cross-entropy loss function
        self.loss_func = nn.BCELoss()

    def forward(self, x):
        # Flatten the input image
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
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
