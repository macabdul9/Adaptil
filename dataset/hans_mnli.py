import torch
from datasets import load_dataset, concatenate_datasets
import numpy as np

def hans_mnli_loaders(config, tokenizer, max_len=256):

    domains = config['domains']

    # which label has least number of samples in train data as well as (test)validation data in all domains
    train_label_dist = 15000 # manually checked
    test_label_dist =  6453 # manually checked

    hans = load_dataset("hans")
    mnli = load_dataset("multi_nli")

    mnli = mnli.remove_columns(['pairID', 'promptID', 'premise_binary_parse', 'premise_parse', 'hypothesis_binary_parse', 'hypothesis_parse'])  # remove the unrelated fields
    mnli = mnli.filter(lambda example:example['label']!=1)  # remove examples having neutral label
    mnli['validation_matched'] = concatenate_datasets(dsets=[mnli['validation_matched'], mnli['validation_mismatched']])

    datasets = {
        "hans":hans,
        "mnli":mnli
    }

    labels = list(set(hans['train']['label']))

    domain_dsets = {}
    for domain in domains:

        domain_dsets[domain] = {
            "train":[]
        }
        domain_dsets[domain].update({"test":[]})

    # take equal numbe of samples for each domain for each label for each set
    for label in labels:
        for domain in domain_dsets:

            if domain == "mnli" and label == 1:
                label = 2  # we don't have label 1 incase of mnli as it means "neutral"

            train = datasets[domain]['train'].filter(lambda example:example['label']==label).shuffle(seed=42).select(range(train_label_dist))

            if domain == "mnli":
                test = datasets[domain]['validation_matched'].filter(lambda example:example['label']==label).shuffle(seed=42).select(range(test_label_dist))
            else:
                test = datasets[domain]['validation'].filter(lambda example:example['label']==label).shuffle(seed=42).select(range(test_label_dist))

            domain_dsets[domain]['train'].append(train)
            domain_dsets[domain]['test'].append(test)


    # concatenate the dataset and shuffle them
    # before it would be list of datasets and after it will be single dataset
    for domain in domains:

        # concate class label datasets
        domain_dsets[domain]['train'] = concatenate_datasets(dsets=domain_dsets[domain]['train']).shuffle(seed=42)
        domain_dsets[domain]['test'] = concatenate_datasets(dsets=domain_dsets[domain]['test']).shuffle(seed=42)

        # # tokenize
        domain_dsets[domain]['train'] = domain_dsets[domain]['train'].map(lambda x: tokenizer(x['premise'], x['hypothesis'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
        domain_dsets[domain]['test'] = domain_dsets[domain]['test'].map(lambda x: tokenizer(x['premise'], x['hypothesis'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)

        # change the dtype
        domain_dsets[domain]['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        domain_dsets[domain]['test'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        # create dataloaders
        domain_dsets[domain]['train'] = torch.utils.data.DataLoader(dataset = domain_dsets[domain]['train'], batch_size=config["batch_size"], shuffle=True, num_workers=4)
        domain_dsets[domain]['test'] = torch.utils.data.DataLoader(dataset = domain_dsets[domain]['test'], batch_size=config["batch_size"], shuffle=True, num_workers=4)
        domain_dsets[domain]['valid'] = domain_dsets[domain]['test'] # validation and test will be same


    # why the hell I am doing this?
    # we all are seeking the same answer from our lives.
    loaders = domain_dsets
    return loaders



