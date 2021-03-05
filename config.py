
config = {
    "tasks":{
        "sa":{
            "num_classes":2,
            "domains":["books", "dvd", "electronics", "kitchen_housewares"],
            "lr":2e-5,
            "batch_size":32,
            "epoch":4,

        },
        "mnli":{
            "num_classes":3,
            "domains":['government', 'telephone', 'fiction', 'travel', 'slate'],
            "lr":2e-5,
            "batch_size":32,
            "epoch":5,
        },

    },
    "models":['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'distilroberta-base'],
    
    "max_seq_length": 128,
    "num_workers":4,

    # training,

    "callback_config":{
        "monitor":"val_acc",
        "min_delta":0.001,
        "patience":2,
        "precision":32,
        "project":"adaptil",
    }
    

}
