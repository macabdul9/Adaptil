
from config import config
from transformers import AutoTokenizer
from models.ContextAwareDAC import ContextAwareDAC
from models.SpeakerClassifier import SpeakerClassifierModel
from Trainer import LightningModel
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import os
import torch


if __name__=="__main__":

    # create the checkpoints dir
    path = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.isdir(path):
        os.mkdir(path)

        
    logger = WandbLogger(
        name="speaker-classifier",
        save_dir=config["save_dir"],
        project=config["project"],
        log_model=True,
    )
    early_stopping = EarlyStopping(
        monitor=config["monitor"],
        min_delta=config["min_delta"],
        patience=5,
    )
    checkpoints = ModelCheckpoint(
        filepath=config["filepath"],
        monitor=config["monitor"],
        save_top_k=1
    )

    classifier = SpeakerClassifierModel(config=config)
    
    model = LightningModel(model=classifier, config=config)
    
    trainer = pl.Trainer(
        # logger=logger,
        # gpus=[0],
        # checkpoint_callback=checkpoints,
        # callbacks=[early_stopping],
        # default_root_dir="./models/",
        # max_epochs=config["epochs"],
        # precision=config["precision"],
        # automatic_optimization=True
    )
    
    
    # trainer.fit(model)
    
    classifier.state_dict()
    
    trainer.test(model)
    
    
    # # save the model 
    # checkpoint = {
    #     'model': SpeakerClassifierModel(config=config),
    #     'state_dict': classifier.state_dict(),
    # }

    # torch.save(checkpoint, 'speaker_classifier.ckpt') 
    # torch.save(classifier, "speaker_classifier.pth")
    
    trainer = pl.Trainer()
    
    tasks = config['tasks'].keys()
    
    models = config['models']
    
    for task in tasks:
        
        for model_name in models:
            
            lm = LightningModel(model_name=model_name)
            
            loaders = create_loaders(task=laoder)
            
            for source in loaders:
                train_laoder, valid_loader = loaders[source]['train'], loaders[source]['valid']
                trainer.fit(lm, train_laoder, valid_loader)
                for target in loaders:
                    results = trainer.test(
                        lm, 
                        loader[target]['valid']
                    )
            
                
            
            
            
    