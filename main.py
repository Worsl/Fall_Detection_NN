"""
The entry script to train the model for fall detection with PyTorch Lightning

@Author: Tang Yuting
@Date 30 Oct 2023
"""

import os
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

IS_DEBUG_MODE = True  # Trainer will only run 1 step on training and testing if set to True
FRAMES_DIRECTORY = 'data/Frames_Extracted'
DEFAULT_ROOT_DIR = 'checkpoint'
IS_ADD_EARLY_STOPPING = True
MAX_EPOCHS = 10


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
    model.load_from_checkpoint(model_checkpoint_hook.best_model_path)
    # Test the model's performance
    trainer.test(model=model, dataloaders=test_loader)


def main():
    # Load image file paths
    train_frames, valid_frames, test_frames = load_image_file_paths(FRAMES_DIRECTORY)

    # Train and test the ResNet model
    resnet_model = BinaryClassificationDetectionModel(base_model='resnet')
    train_set = FallDetectionDataset(train_frames, transform='augmented')  # only apply data augmentation on train set
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_set = FallDetectionDataset(valid_frames, transform='default')
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)
    test_set = FallDetectionDataset(test_frames, transform='default')
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    train_and_test_model(resnet_model, train_loader, valid_loader, test_loader, model_name="ResNet")
    print("ResNet successful ran")

    # Train and test the VGG model

    sfd_model = BinaryClassificationDetectionModel(base_model='vgg16')
    train_set_densenet = FallDetectionDataset(train_frames, transform='default')
    train_loader_densenet = DataLoader(train_set_densenet, batch_size=32, shuffle=True)
    train_and_test_model(sfd_model, train_loader_densenet, test_frames, "sophisticated-fall-detection")
    print("VGGModel successful ran")


# # Train and test the Transformer model
#     simple_model = SimpleFallDetectionModel()  
#     train_set_simple = FallDetectionDataset(train_frames, transform='default')
#     train_loader_simple = DataLoader(train_set_simple, batch_size=32, shuffle=True)
#     train_and_test_model(simple_model, train_loader_simple, test_frames, "simple-fall-detection")
#     print("SimpleFallDetectionModel successfully ran")


if __name__ == "__main__":
    main()