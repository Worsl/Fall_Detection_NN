import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random


transform_default = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])


class FallDetectionDataset(Dataset):
    """
    The wrapper class of the dataset. The label of the frame is inferred from the file name, i.e.,
    is_fall = 'unfall' not in self.frame_files[idx]
    Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4814805/
    """
    TEST_SET_PREFIX = ['ADL2', 'ADL16', 'ADL10', 'ADL17', 'Fall48', 'Fall40', 'Fall15', 'Fall12', 'Fall5', 'Fall1',
                       'Fall51', 'Fall24', 'Fall50', 'Fall20', 'Fall14']    # pre-generated train-test-split

    def __init__(self, frame_files: list, transform=None):
        """
        The labels of frames are inferred from the file name
        :param frame_files: list, including the readable paths of frames
        :param transform: transforms.Compose or str, 'default' refers to the usage of the default transform
        """
        self.frame_files = frame_files


        # Data Augmentation Flags:
        # These flags control various data augmentation techniques.
        # - useRotation: Applies random rotation to the image.
        # - useColorContrastAugment: Applies random color and contrast adjustments to the image.
        # - useNoiseInjection: Adds random noise to simulate real-world imperfections.
        # - useHorizontalFlip: Applies horizontal flipping to the image.
        # - useShearing : Applies random shear to the image.

        self.useRotation = True
        self.useColorContrastAugment = True
        self.useNoiseInjection = True
        self.useHorizontalFlip = True
        self.useShearing = True

    
        if transform == 'default':
            self.transform = transform_default
        else:
            self.transform = transform

    def apply_rotation(self, frame):
        rotation_angle = random.randint(-90, 90)
        return frame.rotate(rotation_angle)  

    def apply_color_contrast_augmentation(self, frame):
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        return transforms.ColorJitter(brightness=brightness_factor, contrast=contrast_factor)(frame)

    def apply_noise_injection(self, frame):
        frame_np = np.array(frame)
        noise = np.random.normal(0, 25, frame_np.shape).astype(np.uint8)
        frame_with_noise = np.clip(frame_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(frame_with_noise)
    
    def apply_horizontal_flip(self,frame):
        # Apply horizontal flipping with a 50% probability
        if random.random() < 0.5:
            frame = frame.transpose(Image.FLIP_LEFT_RIGHT)
        return frame

    def apply_random_shearing(self, frame):
        max_shear_x = 10  # Maximum shear angle in the x-direction (degrees)
        max_shear_y = 10  # Maximum shear angle in the y-direction (degrees)

        shear_x = random.uniform(-max_shear_x, max_shear_x)
        shear_y = random.uniform(-max_shear_y, max_shear_y)

        # Apply shear using Pillow's transform function
        return frame.transform(frame.size, Image.AFFINE, (1, shear_x, 0, shear_y, 1, 0))
    

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        frame = Image.open(self.frame_files[idx])
        
        if self.useRotation:
            frame = self.apply_rotation(frame)

        if self.useColorContrastAugment:
            frame = self.apply_color_contrast_augmentation(frame)

        if self.useNoiseInjection:
            frame = self.apply_noise_injection(frame)

        if self.useHorizontalFlip:
            frame = self.apply_horizontal_flip(frame)

        if self.useShearing:
            frame = self.apply_random_shearing(frame)

        frame = self.transform(frame)

        is_fall = 'unfall' not in self.frame_files[idx]
        return frame, is_fall

