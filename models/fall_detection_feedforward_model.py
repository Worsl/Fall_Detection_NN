import torch
from torch import nn
import pytorch_lightning as pl

class FeedForwardFallDetectionModel(pl.LightningModule):
    """
    A feedforward neural network for binary classification of fall detection.
    """
    def __init__(self, input_size, hidden_size, num_hidden_layers):
        super().__init()
        
        # Define the feedforward layers
        layers = []
        in_features = input_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        
        # Output layer for binary classification
        layers.append(nn.Linear(in_features, 1))
        self.ffn = nn.Sequential(*layers)
        
        # Activation function
        self.activation_func = nn.Sigmoid()

        # Binary cross-entropy loss function
        self.loss_func = nn.BCELoss()

    def forward(self, x):
        # Forward pass through the feedforward layers
        return self.ffn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log("test_loss", loss)
        return loss
