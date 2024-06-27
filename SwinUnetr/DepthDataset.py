import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
from PIL import Image
import albumentations as A
from torchvision import transforms as T


class Depth_dataset(Dataset):
    def __init__(self, root_directory, train=True, transformations=None, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        """
        Dataset class for Monocular Depth Estimation
        :param root_directory: Root path to the dataset with file structure root>>[test.csv,train.csv] with file paths
        :param train: Whether the dataset is train for test, if Train==True, dataset is train else Test
        :param transformations: Augmentations that need to used with the dataset
        """
        self.root_directory = root_directory
        self.dataset_type = train
        self.augmentations = transformations
        self.mean = mean
        self.std = std
        self.normalize_image = T.Compose([T.Normalize(self.mean, self.std)])
        self.standard_transformations = T.Compose(
            [T.ToTensor()])

        # Hard coded intentionally
        if train:
            self.image_paths = list(pd.read_csv(f"{self.root_directory}/train.csv")["Images"])
            self.depth_maps = list(pd.read_csv(f"{self.root_directory}/train.csv")["Depth_maps"])
        else:
            self.image_paths = list(pd.read_csv(f"{self.root_directory}/test.csv")["Images"])
            self.depth_maps = list(pd.read_csv(f"{self.root_directory}/test.csv")["Depth_maps"])

    def __getitem__(self, index):
        """
        Return the ith item of the dataset
        :param index: index of the element to retrieve
        :return: TorchTensor:Image, TorchTensor:depth_map
        """
        image = cv2.imread(self.image_paths[index])
        depth_map = cv2.imread(self.depth_maps[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)

        if self.augmentations is not None:
            aug = self.augmentations(image=image, mask=depth_map)
            image = self.standard_transformations(aug['image'])
            depth_map = self.standard_transformations(aug['mask'])
        else:
            image = self.standard_transformations(image)
            depth_map = self.standard_transformations(depth_map)

        return self.normalize_image(image), depth_map

    def __len__(self):
        """
        Return the length of the respective dataset
        return: Dataset length/Number of images in said dataset
        """
        return len(self.image_paths)


if __name__ == "__main__":
    dataset_path = "../../../Desktop/Renders/"
    test_dataset = Depth_dataset(dataset_path, train=False)
    print(test_dataset.__len__())
    print((test_dataset.depth_maps))
