import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
    VALID_SET_PREFIX = ['ADL3', 'ADL14', 'Fall31', 'Fall23', 'Fall2', 'Fall47', 'Fall22']   # pre-generated

    def __init__(self, frame_files: list, transform=None):
        """
        The labels of frames are inferred from the file name
        :param frame_files: list, including the readable paths of frames
        :param transform: transforms.Compose or str, 'default' refers to the usage of the default transform
        """
        self.frame_files = frame_files
        if transform == 'default':
            self.transform = transform_default
        else:
            self.transform = transform

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        frame = Image.open(self.frame_files[idx])
        frame = self.transform(frame)
        is_fall = 'unfall' not in self.frame_files[idx]
        return frame, is_fall