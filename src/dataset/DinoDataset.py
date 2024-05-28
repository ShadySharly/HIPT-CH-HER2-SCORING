from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import torchvision
class DinoDataset(Dataset):
    def __init__(
        self,
        partitions_src: str,
        split: str,
        set_type: str,
        images_folder: str,
        **kwargs,
    ) -> None:

        self.partitions = pd.read_parquet(partitions_src)
        self.images_folder = images_folder
        self.partition_used = self.partitions.loc[
            self.partitions.partition
            == (f"{set_type}_{split}" if set_type != "test" else "test")
        ]
        self.labels = self.partition_used.new_codigo.values
        self.images = self.partition_used.ImageId.values
        self.transform = transforms.Compose([ torchvision.transforms.PILToTensor()])
        

    def __getitem__(self, index):

        data_dict = {
            "data": self.transform(Image.open(f"{self.images_folder}/{self.images[index]}")),
            "label": self.labels[index],
        }

        return data_dict, data_dict["label"]

    def __len__(self):
        return len(self.labels)
