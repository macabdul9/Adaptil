import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from config import config


def create_logger(project, name):

    logger = WandbLogger(
        name=name,
        project=project,
        save_dir=os.getcwd(),
        log_model=True,
    )
    return logger

def create_early_stopping_and_model_checkpoint(callback_config, path, run_name):

    task, model_name, source = run_name.split("_")

    early_stopping = EarlyStopping(
        monitor=callback_config["monitor"],
        min_delta=callback_config["min_delta"],
        patience=callback_config['patience'],
    )

    checkpoints = ModelCheckpoint(
        filename=os.path.join(path, source+".ckpt"),
        monitor=callback_config["monitor"],
        save_top_k=1,
        verbose=True,
    )

    return early_stopping, checkpoints

def create_trainer(callback_config, run_name, path):

    task, model_name, source = run_name.split("-")


    logger = create_logger(project=callback_config['project'], name=run_name)

    early_stopping, checkpoints = create_early_stopping_and_model_checkpoint(callback_config, path, run_name)

    trainer = pl.Trainer(
        logger=logger,
        gpus=[0],
        checkpoint_callback=checkpoints,
        callbacks=[early_stopping],
        max_epochs=config['tasks'][task]["epochs"],
        precision=config["precision"],
        automatic_optimization=True
    )

    return trainer
