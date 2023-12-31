"""
The entry script to train the model for fall detection with PyTorch Lightning

@Author: Tang Yuting
@Date 30 Oct 2023
"""

import os
import fire
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

# Import custom modules
# Import all models here
from dataset import FallDetectionDataset

from models.baseline import BinaryClassificationDetectionModel
from image_augmentation import DataAugmentation

IS_DEBUG_MODE = False  # Trainer will only run 1 step on training and testing if set to True
DEFAULT_ROOT_DIR = 'checkpoint'
IS_ADD_EARLY_STOPPING = True
MAX_EPOCHS = 50


def load_image_file_paths(frames_directory: str):
    """
    Loads train, valid, and test images from `frames_directory` without data augmentation.
    :param frames_directory: str
    :return: tuple, (train_frames, valid_frames, test_frames)
    """
    total_frames = os.listdir(frames_directory)
    train_frames = []
    test_frames = []
    valid_frames = []

    for frame in tqdm(total_frames, desc=f'Loading images from the directory {frames_directory}'):
        scenario_name = frame.split('_')[0]
        if '.jpg' not in frame:
            continue

        frame = os.path.join(frames_directory, frame)

        if scenario_name in FallDetectionDataset.TEST_SET_PREFIX:
            test_frames.append(frame)
        elif scenario_name in FallDetectionDataset.VALID_SET_PREFIX:
            valid_frames.append(frame)
        else:
            train_frames.append(frame)

    print(f"len(train_frames) = {len(train_frames)}\tlen(valid_frames) = {len(valid_frames)}\t"
          f"len(test_frames) = {len(test_frames)}")
    return train_frames, valid_frames, test_frames


def train_and_test_model(model, train_loader, valid_loader, test_loader, model_name: str = ""):
    """
    Initialize the trainer. Then, train and test the model.
    :param model: pl.LightningModule
    :param train_loader: torch.utils.data.DataLoader
    :param valid_loader: torch.utils.data.DataLoader
    :param test_loader: torch.utils.data.DataLoader
    :param model_name: str, for logging purpose
    :return:
    """
    # Initialize a logging tool (WandbLogger)
    logger = WandbLogger(save_dir='.', project='fall_detection', log_model=True)
    logger.experiment.config["model"] = model_name
    logger.experiment.config["learning_rate"] = model.lr
    logger.experiment.config["is_pretrained"] = model.is_pretrained

    callbacks = []
    model_checkpoint_hook = ModelCheckpoint(monitor='valid_loss', mode='min', save_top_k=1)
    callbacks.append(model_checkpoint_hook)
    if IS_ADD_EARLY_STOPPING:
        callbacks.append(EarlyStopping(monitor="valid_loss", mode="min", patience=3))

    # Initialize a PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, enable_checkpointing=True, logger=logger,
                         enable_progress_bar=True, fast_dev_run=IS_DEBUG_MODE, default_root_dir=DEFAULT_ROOT_DIR,
                         callbacks=callbacks)
    
    # Train the model on the training dataset
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    # Load the best model after training, ref: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.
    # ModelCheckpoint.html
    if not IS_DEBUG_MODE:   # debug mode does not save any checkpoint, so the steps below are skipped.
        print(f'Loading checkpoint from {model_checkpoint_hook.best_model_path}')
        BinaryClassificationDetectionModel.load_from_checkpoint(model_checkpoint_hook.best_model_path, strict=False)
    # Test the model's performance
    trainer.test(model=model, dataloaders=test_loader)


def main(model_name: str = 'resnet', learning_rate: float = 1e-5, is_pretrained: bool = True, batch_size: int = 32,
         frames_directory: str = 'data/Frames_Extracted_Camera2', is_extra_fc_layers: bool = False,
         is_freeze_base_model: bool = False):
    # Load image file paths
    train_frames, valid_frames, test_frames = load_image_file_paths(frames_directory)

    # Train and test the ResNet model
    resnet_model = BinaryClassificationDetectionModel(base_model=model_name, learning_rate=learning_rate,
                                                      is_pretrained=is_pretrained,
                                                      is_extra_fc_layers=is_extra_fc_layers,
                                                      is_freeze_base_model=is_freeze_base_model)
    train_set = FallDetectionDataset(train_frames, transform='augmented')  # only apply data augmentation on train set
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_set = FallDetectionDataset(valid_frames, transform='default')
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_set = FallDetectionDataset(test_frames, transform='default')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    train_and_test_model(resnet_model, train_loader, valid_loader, test_loader, model_name)


if __name__ == "__main__":
    fire.Fire(main)
