
from models import DinoModel
from functions.train import train_loop_model
from functions.test import test_loop_model
import torch


if __name__ == "__main__":

    kwargs = {
        "checkpoint_dino_one": "./checkpoints/vit256_small_dino.pth",
        "checkpoint_dino_two": "./checkpoints/vit4k_xs_dino.pth",
        "device_one": torch.device("cuda:0"),
        "device_two": torch.device("cuda:0"),
    }
    model = DinoModel(**kwargs)
    print(model)
    print("Working !!!")