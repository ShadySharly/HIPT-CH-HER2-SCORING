
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


    partitions = pd.read_parquet("/home/bgamboa/side_projects/HIPT-CH-HER2-SCORING/src/data/partitions_dinosharly.parquet")
    tr_examples = partitions.loc[partitions.partition == "training_0"][["ImageId", "new_codigo"]].values
    vl_examples = partitions.loc[partitions.partition == "validation_0"][["ImageId", "new_codigo"]].values 
    ts_examples = partitions.loc[partitions.partition == "test"][["ImageId", "new_codigo"]].values 
    
    # edit this
    src_folder = "/home/bgamboa/side_projects/HIPT-CH-HER2-SCORING/src/data/image"
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    im_tr, lb_tr = tr_examples[:, 0], torch.from_numpy(tr_examples[:,1].astype(int))
    im_vl, lb_vl = vl_examples[:, 0], torch.from_numpy(vl_examples[:,1].astype(int))
    im_ts, lb_ts = ts_examples[:, 0], torch.from_numpy(ts_examples[:,1].astype(int))


    for epoch in range(n_epochs):
        model.train()
        logging.info(f"Starting Epoch {epoch}")
        for i, (imp, label) in enumerate(zip(im_tr, lb_tr)):
            abs_imp_path = f"{src_folder}/{imp}"
            image_input = Image.open(abs_imp_path)
            y_pred = model.forward(**{"x": image_input})
            y_true = label.long()[None, ...].to(device="cuda:0")
            loss = F.cross_entropy(y_pred, y_true)
            loss =  loss / n_cum_grads
            loss.backward()
            if ((i+1) % n_cum_grads) == 0 or (i + 1 == len(im_tr)):
                logging.info(f"Optimizer step ... {i}")
                optimizer.step()
                optimizer.zero_grad()
                logging.info("Epoch end, evaluation ...")
        with torch.no_grad():
            pred = []
            true = []
            toks = []
            model.eval()
            for j, (imp, label) in enumerate(zip(im_vl, lb_vl)):
                abs_imp_path = f"{src_folder}/{imp}"
                image_input = Image.open(abs_imp_path)
                y_pred = model.forward(**{"x": image_input}).to(device="cpu")
                tokens = model.encode(**{"x": image_input}).to(device="cpu")
                y_true = label.long()[None, ...].to(device="cpu")
                pred.append(y_pred)
                true.append(y_true)
                toks.append(tokens)
                
            pred = torch.cat(pred, axis =0).numpy()
            true = torch.cat(true, axis =0).numpy().astype(int)
            tokens = torch.cat(toks, axis =0).numpy()
            pred_ = np.argmax(pred, axis=1).astype(int)
            true_ = true
            where_wrong = (pred_==true_)
            logging.info(f"Bad classifications {true[~where_wrong]}")
            logging.info(pred_==true_)
            logging.info(np.sum(pred_==true_))
            acc = np.sum(pred_==true_) /len(true_)
            logging.info(f"Accuracy {acc} epoch {epoch}")
            handler_pickle(tokens, './outputs', 'tokens_dino_epoch_{}'.format(epoch))
            handler_pickle(true, './outputs', 'trues_dino_epoch_{}'.format(epoch))


        
