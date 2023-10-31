"""
This script includes a baseline classifier that uses pre-trained ResNet50
"""


from sklearn.metrics import roc_auc_score
import torchvision
import torch
from torch import nn
import pytorch_lightning as pl


class ResNetFallDetectionModel(pl.LightningModule):
    """
    A baseline binary classifier that uses pre-trained ResNet50 as the feature extractor and
    a linear layer as the classifier
    """
    def __init__(self):
        super().__init__()
        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        self.fc = nn.Linear(1000, 1)
        self.activation_func = nn.Sigmoid()

        # loss function
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
        Process each batch during test. The evaluation metrics include binary cross entropy loss and ROC-AUC
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
