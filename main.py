"""
The entry script to train the model for fall detection with PyTorch Lightning

@Author: Tang Yuting
@Date 30 Oct 2023
"""


import os
import cv2
import torch
from tqdm.auto import tqdm

from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from dataset import FallDetectionDataset
from models.baseline import ResNetFallDetectionModel


FRAMES_DIRECTORY = 'data/Frames_Extracted'
IS_DEBUG_MODE = False  # Trainer will only run 1 step on training and testing if set to True
MAX_EPOCH_NUM = 10

assert MAX_EPOCH_NUM >= 1


# load image file paths
total_frames = os.listdir(FRAMES_DIRECTORY)
train_frames = []
test_frames = []
for frame in tqdm(total_frames, desc=f'Loading images from the directory {FRAMES_DIRECTORY}'):
    scenario_name = frame.split('_')[0]
    if '.jpg' not in frame:
        continue
    frame = os.path.join(FRAMES_DIRECTORY, frame)
    if scenario_name in FallDetectionDataset.TEST_SET_PREFIX:
        test_frames.append(frame)
    else:
        train_frames.append(frame)


train_set = FallDetectionDataset(train_frames, transform='default')
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)


logger = WandbLogger(save_dir='.', name="fall-detection")


model = ResNetFallDetectionModel()
trainer = pl.Trainer(max_epochs=MAX_EPOCH_NUM, enable_checkpointing=True, logger=logger,
                     enable_progress_bar=True, fast_dev_run=IS_DEBUG_MODE)
trainer.fit(model, train_dataloaders=train_loader)


# inference on test set
if len(test_frames) > 0:
    test_set = FallDetectionDataset(test_frames, transform='default')
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    trainer.test(model=model, dataloaders=test_loader)
