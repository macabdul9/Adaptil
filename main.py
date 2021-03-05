import os
import torch
import gc
import pandas as pd

from config import config
from dataset.dataset import create_loaders
from evaluation import evaluate
from utils import *

import pytorch_lightning as pl
from transformers import AutoTokenizer
from Trainer import LightningModel
import torch

if __name__=="__main__":

    task = "sa"  # define your task here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in model_list:

        tokenizer = AutoTokenizer.from_pretrained(model_name, usefast=True, use_lower_case=True)
        loaders = create_loaders(task=task, tokenizer=tokenizer)

        for source in loaders:

            lm = LightningModel(model_name=model_name, config=config)

            path = os.path.join(os.getcwd(), "outputs", task, model_name, source)

            trainer_config = {
                "callback_config": config['callback_config'],
                "path": path,
                "device": device,
            }

            train_loader, valid_loader = loaders[source]['train'], loaders[source]['valid']

            trainer.fit(lm, train_loader, valid_loader)

            for target in loaders:
                # results = trainer.test(
                #     lm,
                #     loaders[target]['valid']
                # )
                true_label, pred_label, report  = evaluate(model=lm, loader=loaders[target]['valid'], device=device)  # handle source corner case

                if(target == source):
                    # append source with name

            # save json for that domain here





