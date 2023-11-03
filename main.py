"""
The entry script to train the model for fall detection with PyTorch Lightning

@Author: Tang Yuting
@Date 30 Oct 2023
"""

import os
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl

# Import custom modules
# Import all models here
from dataset import FallDetectionDataset
from models.baseline import ResNetFallDetectionModel
from models.VGG_model import VGGFallDetectionModel
from models.simple_fall_detection_model import SimpleFallDetectionModel
from image_augmentation import DataAugmentation

IS_DEBUG_MODE = True  # Trainer will only run 1 step on training and testing if set to True
FRAMES_DIRECTORY = 'data/Frames_Extracted'


def load_image_file_paths(frames_directory):
    total_frames = os.listdir(frames_directory)
    train_frames = []
    test_frames = []

    dataAugmentation = DataAugmentation()

    for frame in tqdm(total_frames, desc=f'Loading images from the directory {frames_directory}'):
        scenario_name = frame.split('_')[0]
        if '.jpg' not in frame:
            continue

        frame = os.path.join(frames_directory, frame)

        if scenario_name in FallDetectionDataset.TEST_SET_PREFIX:
            test_frames.append(frame)

        else:
            frame = dataAugmentation.start_augment(frame)
            train_frames.append(frame)
    
    return train_frames, test_frames


def train_and_test_model(model, train_loader, test_frames, name):
    # Initialize a logging tool (WandbLogger)
    logger = WandbLogger(save_dir='.', name=name)
    
    # Initialize a PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=10, enable_checkpointing=True, logger=logger,
                         enable_progress_bar=True, fast_dev_run=IS_DEBUG_MODE)
    
    # Train the model on the training dataset
    trainer.fit(model, train_dataloaders=train_loader)
    
    # Inference on the test set (if available)
    if len(test_frames) > 0:
        test_set = FallDetectionDataset(test_frames, transform='default')
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        trainer.test(model=model, dataloaders=test_loader)



def main():

    # Load image file paths
    train_frames, test_frames = load_image_file_paths(FRAMES_DIRECTORY)
    

    # Train and test the ResNet model
    resnet_model = ResNetFallDetectionModel()
    train_set_resnet = FallDetectionDataset(train_frames, transform='default')
    train_loader_resnet = DataLoader(train_set_resnet, batch_size=32, shuffle=True)
    train_and_test_model(resnet_model, train_loader_resnet, test_frames, "fall-detection")
    print("ResNetFallDetectionModel successful ran")


    # Train and test the VGG model
    VGG_model = VGGFallDetectionModel()
    train_set_densenet = FallDetectionDataset(train_frames, transform='default')
    train_loader_densenet = DataLoader(train_set_densenet, batch_size=32, shuffle=True)
    train_and_test_model(VGG_model, train_loader_densenet, test_frames, "sophisticated-fall-detection")
    print("VGGFallDetectionModel successful ran")


# Train and test the Transformer model
    simple_model = SimpleFallDetectionModel()  
    train_set_simple = FallDetectionDataset(train_frames, transform='default')
    train_loader_simple = DataLoader(train_set_simple, batch_size=32, shuffle=True)
    train_and_test_model(simple_model, train_loader_simple, test_frames, "simple-fall-detection")
    print("SimpleFallDetectionModel successfully ran")


if __name__ == "__main__":
    main()