"""
This script includes a baseline classifier that uses pre-trained ResNet50
"""

import wandb
import torchvision
import torch
import torchmetrics
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

        # evaluation metrics
        self.pr_curve = torchmetrics.PrecisionRecallCurve(task="binary")
        self.confusion_matrix_calculator = torchmetrics.ConfusionMatrix(task="binary", num_classes=2)
        self.auroc = torchmetrics.AUROC(task="binary")

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
        self.log("test_loss", loss, on_epoch=True)  # on_epoch=True to log the epoch average
        return y_hat.detach(), y

    def test_epoch_end(self, outputs):
        pred = [x[0] for x in outputs]
        pred = torch.cat(pred)
        target = [x[1] for x in outputs]
        target = torch.cat(target).int()
        # get precision-recall curve
        precisions, recalls, _ = self.pr_curve(pred, target)
        pr_curve_data = list(zip(recalls.cpu().tolist(), precisions.cpu().tolist()))
        table = wandb.Table(data=pr_curve_data, columns=["Recall", "Precision"])
        wandb.log(
                {"pr_curve": wandb.plot.line(table, "Recall", "Precision",
                                             title="Precision-Recall Curve")})
        # get auc
        auc_value = self.auroc(pred, target)
        wandb.log({"auc": auc_value})
        # get confusion matrix
        wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                           y_true=target.cpu().flatten().tolist(),
                                                           preds=(pred > 0.5).cpu().flatten().tolist(),
                                                           class_names=['is_fall', 'not_fall'])})
