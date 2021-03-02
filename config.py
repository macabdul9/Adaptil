

config = {
    "tasks":{
        "mnli":{
            "num_classes":3,
            "domains":['government', 'telephone', 'fiction', 'travel', 'slate'],
        },
        "sa":{
            "text_field":"review_text",
            "label_field":"sentiment",
            "num_classes":2,
            "domains":["books", "dvd", "electronics", "kitchen_housewares"]
        }
        
    }
}