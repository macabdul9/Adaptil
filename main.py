import os
import gc
import json
import torch
import argparse

from config import config
from dataset.dataset import create_loaders
from evaluation import evaluate
from utils import *

import pytorch_lightning as pl
from transformers import AutoTokenizer
from Trainer import LightningModel

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--Task", help="'sa' for sentiment analysis, 'mnli' for multi_nli")

    args = parser.parse_args()

    task = args.Task  # define your task here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for model_name in model_list:

        tokenizer = AutoTokenizer.from_pretrained(model_name, usefast=True, use_lower_case=True)
        loaders = create_loaders(task=task, tokenizer=tokenizer)

        for source in loaders:

            if(len(loaders[source])!=2):
                continue

            lm = LightningModel(model_name=model_name, config=config['tasks'][task])

            # create the checkpoint path
            PATH = os.path.join(os.getcwd(), "outputs", task, model_name)
            os.makedirs(PATH, exist_ok=True)

            run_name = task+"-"+model_name +"-"+source

            trainer = create_trainer(callback_config=config['callback_config'], path=PATH, run_name=run_name)

            train_loader, valid_loader = loaders[source]['train'], loaders[source]['valid']

            trainer.fit(lm, train_loader, valid_loader)

            # load best checkpoint
            lm.load_from_checkpoint(PATH)

            for target in loaders:

                f1, accuracy  = evaluate(model=lm, loader=loaders[target]['valid'], device=device)
                # save into results
                results[model_name] = {
                    source:{
                        target:{
                            "f1":f1,
                            "accuracy":accuracy
                        }
                    }
                }

            # delete model
            del lm
            gc.collect()
            torch.cuda.empty_cache()


    # save the results into json file at outputs/
    with open(os.path.join(os.getcwd(), "outputs", "results.json"), "w") as file:
        json.dump(results, file)