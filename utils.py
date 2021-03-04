import os 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def create_logger(project, name, save_dir):
    
    logger = WandbLogger(
        name=name,
        project=project,
        save_dir=os.getcwd(),
        log_model=True,
    )
    return logger

def create_early_stopping_and_model_checkpoint(callback_config):
    early_stopping = EarlyStopping(
        monitor=callback_config["monitor"],
        min_delta=callback_config["min_delta"],
        patience=callback_config['patience'],
    )
    checkpoints = ModelCheckpoint(
        filepath=callback_config["filepath"],
        monitor=callback_config["monitor"],
        save_top_k=1
    )
    
    return early_stopping, checkpoints

def create_trainer(config, run_name, path):
    
    logger = create_logger(project=project, name=name)
    
    early_stopping, checkpoints = create_early_stopping_and_model_checkpoint(config['callback_config'])
    
    trainer = pl.Trainer(
        logger=logger,
        gpus=[0],
        checkpoint_callback=checkpoints,
        callbacks=[early_stopping],
        default_root_dir="./models/",
        max_epochs=config["epochs"],
        precision=config["precision"],
        automatic_optimization=True
    )
    
    return trainer
    