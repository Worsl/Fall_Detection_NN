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
from models.baseline import BinaryClassificationDetectionModel


IS_DEBUG_MODE = True  # Trainer will only run 1 step on training and testing if set to True
FRAMES_DIRECTORY = 'data/Frames_Extracted'
DEFAULT_ROOT_DIR = 'checkpoint'


def load_image_file_paths(frames_directory):
    total_frames = os.listdir(frames_directory)
    train_frames = []
    test_frames = []
    
    for frame in tqdm(total_frames, desc=f'Loading images from the directory {frames_directory}'):
        scenario_name = frame.split('_')[0]
        if '.jpg' not in frame:
            continue
        frame = os.path.join(frames_directory, frame)
        if scenario_name in FallDetectionDataset.TEST_SET_PREFIX:
            test_frames.append(frame)
        else:
            train_frames.append(frame)
    
    return train_frames, test_frames


def train_and_test_model(model, train_loader, test_frames, name):
    # Initialize a logging tool (WandbLogger)
    logger = WandbLogger(save_dir='.', name=name)
    
    # Initialize a PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=10, enable_checkpointing=True, logger=logger,
                         enable_progress_bar=True, fast_dev_run=IS_DEBUG_MODE, default_root_dir=DEFAULT_ROOT_DIR)
    
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
    resnet_model = BinaryClassificationDetectionModel(base_model='resnet')
    train_set_resnet = FallDetectionDataset(train_frames, transform='default')
    train_loader_resnet = DataLoader(train_set_resnet, batch_size=32, shuffle=True)
    train_and_test_model(resnet_model, train_loader_resnet, test_frames, "fall-detection")
    print("ResNet successful ran")


    # Train and test the VGG model
    sfd_model = BinaryClassificationDetectionModel(base_model='vgg16')
    train_set_densenet = FallDetectionDataset(train_frames, transform='default')
    train_loader_densenet = DataLoader(train_set_densenet, batch_size=32, shuffle=True)
    train_and_test_model(sfd_model, train_loader_densenet, test_frames, "sophisticated-fall-detection")
    print("VGGModel successful ran")




if __name__ == "__main__":
    main()
