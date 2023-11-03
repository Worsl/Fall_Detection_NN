import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

class DataAugmentation:
    def __init__(self):
        self.useRotation = True
        self.useColorContrastAugment = True
        self.useNoiseInjection = True
        self.useHorizontalFlip = True
        self.useShearing = True

    def start_augment(self, frame_path):
        # Load the image from the file path
        frame = Image.open(frame_path)

        augmentation_choices = []

        if self.useRotation:
            augmentation_choices.append(self.apply_rotation)

        if self.useColorContrastAugment:
            augmentation_choices.append(self.apply_color_contrast_augmentation)

        if self.useNoiseInjection:
            augmentation_choices.append(self.apply_noise_injection)

        if self.useHorizontalFlip:
            augmentation_choices.append(self.apply_horizontal_flip)

        if self.useShearing:
            augmentation_choices.append(self.apply_random_shearing)

        if augmentation_choices:
            selected_augmentation = random.choice(augmentation_choices)
            augmented_frame = selected_augmentation(frame)

            # Overwrite the original image with the augmented image
            augmented_frame.save(frame_path)

    def apply_rotation(self, frame):
        rotation_angle = random.randint(-15, 15)
        return frame.rotate(rotation_angle)

    def apply_color_contrast_augmentation(self, frame):
        brightness_factor = random.uniform(0.9, 1.1)
        contrast_factor = random.uniform(0.9, 1.1)
        enhancer = ImageEnhance.Brightness(frame)
        frame = enhancer.enhance(brightness_factor)
        enhancer = ImageEnhance.Contrast(frame)
        frame = enhancer.enhance(contrast_factor)
        return frame

    def apply_noise_injection(self, frame):
        frame_np = np.array(frame)
        noise = np.random.normal(0, 1, frame_np.shape).astype(np.uint8)
        frame_with_noise = np.clip(frame_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(frame_with_noise)

    def apply_horizontal_flip(self, frame):
        frame = ImageOps.mirror(frame)
        return frame


    def apply_random_shearing(self,frame):
        max_shear_x = 0.5  # Maximum shear angle in the x-direction (degrees)
        max_shear_y = 0.5  # Maximum shear angle in the y-direction (degrees)

        shear_x = random.uniform(-max_shear_x, max_shear_x)
        shear_y = random.uniform(-max_shear_y, max_shear_y)

        return frame.transform(frame.size, Image.AFFINE, (1, shear_x, 0, shear_y, 1, 0))
