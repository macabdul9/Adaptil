
import torch
import os

config = {
    "tasks":{
        "sa":{
            "num_classes":2,
            "domains":["books", "dvd", "electronics", "kitchen_housewares"]
        },
        "mnli":{
            "num_classes":3,
            "domains":['government', 'telephone', 'fiction', 'travel', 'slate'],
        },

    },
    "models":['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'distilroberta-base'],
    "batch_size": 64,
    "max_seq_length": 512,
    "num_workers":4,
    
    # training
    
    
    "lr":1e-5,
    "average":"micro",
    
    "training_config":{
        "monitor":"val_f1",
        "min_delta":0.001,
        "precision":32,
        "project":"adaptil",
    }
    
}
