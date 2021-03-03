config = {
    "tasks":{
        "sa":{
            "text_field":"review_text",
            "label_field":"sentiment",
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
}