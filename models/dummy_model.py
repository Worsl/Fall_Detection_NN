import torch
from torch import nn
import pytorch_lightning as pl
import torchvision

class VGGFallDetectionModel(pl.LightningModule):
    """
    Another binary classifier using VGG16 as the feature extractor, with the same input feature size.
    """
    def __init__(self):
        super().__init__()
        # Define the feature extractor (e.g., VGG16)
        self.feature_extractor = torchvision.models.vgg16(pretrained=True)
        self.fc = nn.Linear(1000, 1)  # Linear layer for binary classification
        self.activation_func = nn.Sigmoid()  # Sigmoid activation

        # Binary cross-entropy loss function
        self.loss_func = nn.BCELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        z = self.feature_extractor(x)
        z = self.fc(z)
        y_hat = self.activation_func(z)
        y_hat = y_hat.squeeze(1)
        loss = nn.functional.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, batch, batch_idx):
        """
        Process each batch during test. The evaluation metrics include binary cross-entropy loss.
        """
        x, y = batch
        y = y.float()
        z = self.feature_extractor(x)
        z = self.fc(z)
        y_hat = self.activation_func(z)
        y_hat = y_hat.squeeze(1)
        loss = nn.functional.binary_cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        return loss
