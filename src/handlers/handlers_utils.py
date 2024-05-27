import torch
from collections import OrderedDict
import pickle
import pandas as pd
import os


def checkpoint_handler(checkpoint_path):
    od = OrderedDict()
    checkpoint = torch.load(checkpoint_path)
    for key in checkpoint["state_dict"].keys():
        od[key.replace("model.", "")] = checkpoint["state_dict"][key]
    return od

def handler_pickle(file, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/{name}.pickle", "wb") as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
