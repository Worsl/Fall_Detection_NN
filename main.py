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
from models.fall_detection_feedforward_model import FeedForwardFallDetectionModel


IS_DEBUG_MODE = False  # Trainer will only run 1 step on training and testing if set to True
FRAMES_DIRECTORY = 'data/Frames_Extracted'



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

    # Train and test the FeedForward model
    input_size = 1000
    hidden_size = 256
    num_hidden_layers = 2
    ffn_model = FeedForwardFallDetectionModel(input_size, hidden_size, num_hidden_layers)
    train_set_ffn = FallDetectionDataset(train_frames, transform='default')
    train_loader_ffn = DataLoader(train_set_ffn, batch_size=32, shuffle=True)
    train_and_test_model(ffn_model, train_loader_ffn, test_frames, "fall-detection-ffn")


if __name__ == "__main__":
    main()
