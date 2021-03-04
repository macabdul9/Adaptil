import os
import torch
import gc 

from config import config
from dataset.dataset import create_loaders

from transformers import AutoTokenizer

from Trainer import LightningModel
import pytorch_lightning as pl
from utils import *
from evaluation import evaluate



    


if __name__=="__main__":
    

    task = "sa"  # define your task here

    for model_name in model_list:


        tokenizer = AutoTokenizer.from_pretrained(model_name, usefast=True, use_lower_case=True)
        
        loaders = create_loaders(task=task, tokenizer=tokenizer)

        for source in loaders:
            
            # we have to create model for each source , fuck its going to take hell lot of time 

            lm = LightningModel(model_name=model_name, config=config)
            
            path = os.path.join(os.getcwd(), "outputs", model_name, source)
            
            trainer_config = {
                "callback_config":config['callback_config'],
                "path":path,
                "device":config["device"]
                
            }
            
            train_loader, valid_loader = loaders[source]['train'], loaders[source]['valid']
            
            trainer.fit(lm, train_loader, valid_loader)
            
            for target in loaders:
                # results = trainer.test(
                #     lm,
                #     loaders[target]['valid']
                # )
                true_label, pred_label, report  = evaluate(model=lm, loader=loaders[target]['valid'], device=config['device'])
                
                # save the result at path
                





