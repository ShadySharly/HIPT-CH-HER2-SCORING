from models import DinoModel
from functions.train import train_loop_model
from functions.test import test_loop_model
from handlers import handler_pickle
import pandas as pd
import torch
import logging
import colorlog
import datetime
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from dataset import DinoDataset
from torchvision import transforms
import torchvision
LOG_FILENAME = f"./logs/main_dino_{datetime.datetime.now()}.log"
logger = logging.getLogger()
logging.root.handlers = []
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s")
)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME, encoding="utf-8"),
        handler,
    ],
)

if __name__ == "__main__":

    n_epochs = 20
    n_cum_grads = 64

    kwargs = {
        "checkpoint_dino_one": "./checkpoints/vit256_small_dino.pth",
        "checkpoint_dino_two": "./checkpoints/vit4k_xs_dino.pth",
        "device_one": torch.device("cuda:0"),
        "device_two": torch.device("cuda:0"),
    }
    model = DinoModel(**kwargs)
    model.to(device="cuda:0")

    logging.info(model)
    logging.info("Working !!!")
    logging.info("Loading dataset ...")
    logging.info("\n\n")

    # edit this
    src_folder = "/home/bgamboa/side_projects/HIPT-CH-HER2-SCORING/src/data/image"
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    data_loader_train = DataLoader(
        DinoDataset(
            **{
                "partitions_src": "/home/bgamboa/side_projects/HIPT-CH-HER2-SCORING/src/data/partitions_dinosharly.parquet",
                "split": "0",
                "set_type": "training",
                "images_folder": "/home/bgamboa/side_projects/HIPT-CH-HER2-SCORING/src/data/image",
          
            }
        ), batch_size= 1, shuffle=True
    )

    data_loader_validation = DataLoader(
        DinoDataset(
            **{
                "partitions_src": "/home/bgamboa/side_projects/HIPT-CH-HER2-SCORING/src/data/partitions_dinosharly.parquet",
                "split": "0",
                "set_type": "validation",
                "images_folder": "/home/bgamboa/side_projects/HIPT-CH-HER2-SCORING/src/data/image",
             
            }
        )
    )

    data_loader_test = DataLoader(
        DinoDataset(
            **{
                "partitions_src": "/home/bgamboa/side_projects/HIPT-CH-HER2-SCORING/src/data/partitions_dinosharly.parquet",
                "split": "0",
                "set_type": "test",
                "images_folder": "/home/bgamboa/side_projects/HIPT-CH-HER2-SCORING/src/data/image",
            
            }
        )
    )


    for epoch in range(n_epochs):
        model.train()
        logging.info(f"Starting Epoch {epoch}")
        for i, (batch_data, label) in enumerate(data_loader_train):
            y_pred = model.forward(**{"x": batch_data["data"]})
            y_true = label.long().to(device="cuda:0")
            loss = F.cross_entropy(y_pred, y_true)
            loss = loss / n_cum_grads
            loss.backward()
            if ((i + 1) % n_cum_grads) == 0 or (i + 1 == len(data_loader_train)):
                logging.info(f"Optimizer step ... {i}")
                optimizer.step()
                optimizer.zero_grad()
                logging.info("Epoch end, evaluation ...")
        with torch.no_grad():
            pred = []
            true = []
            toks = []
            model.eval()
            for j, (batch_data, label) in enumerate(data_loader_validation):
                y_pred = model.forward(**{"x":  batch_data["data"]}).to(device="cpu")
                tokens = model.encode(**{"x":  batch_data["data"]}).to(device="cpu")
                y_true = label.long().to(device="cpu")
                pred.append(y_pred)
                true.append(y_true)
                toks.append(tokens)
            pred = torch.cat(pred, axis=0).numpy()
            true = torch.cat(true, axis=0).numpy().astype(int)
            tokens = torch.cat(toks, axis=0).numpy()
            pred_ = np.argmax(pred, axis=1).astype(int)
            true_ = true
            where_wrong = pred_ == true_
            logging.info(f"Bad classifications {true[~where_wrong]}")
            logging.info(pred_ == true_)
            logging.info(np.sum(pred_ == true_))
            acc = np.sum(pred_ == true_) / len(true_)
            logging.info(f"Accuracy {acc} epoch {epoch}")
            handler_pickle(tokens, "./outputs", "tokens_dino_epoch_{}".format(epoch))
            handler_pickle(true_, "./outputs", "trues_dino_epoch_{}".format(epoch))
