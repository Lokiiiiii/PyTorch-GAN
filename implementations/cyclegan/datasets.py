import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def discover_files(paths):
    files = []
    for p in paths:
        files += glob.glob(p + "/*/*.jpg")
        files += glob.glob(p + "/*/*/*.jpg")
    files = list(set(files))
    random.shuffle(files)
    return files


class ImageDataset:
    TEST_PERCENTAGE = 0.1
    TRAIN_PERCENTAGE = 1-TEST_PERCENTAGE

    def __init__(self, rootA, rootB, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = discover_files(rootA)
        self.files_B = discover_files(rootB)

        self.index_A = int(self.TRAIN_PERCENTAGE*len(self.files_A))
        self.index_B = int(self.TRAIN_PERCENTAGE*len(self.files_B))

        self.train_A = self.files_A[:self.index_A]
        self.train_B = self.files_B[:self.index_B]

        self.test_A = self.files_A[self.index_A:]
        self.test_B = self.files_B[self.index_B:]


    def train(self):
        return ImageDatasetSlice(self.train_A, self.train_B, self.transform, self.unaligned)


    def test(self):
        return ImageDatasetSlice(self.test_A, self.test_B, self.transform, self.unaligned)


class ImageDatasetSlice(Dataset):
    def __init__(self, files_A, files_B, transform, unaligned):
        self.files_A = files_A
        self.files_B = files_B
        self.transform = transform
        self.unaligned = unaligned


    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}


    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
