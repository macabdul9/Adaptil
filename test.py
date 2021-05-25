from transformers import AutoTokenizer
# from dataset.imdb_sst2 import imdb_sst2_loaders
# from dataset.mnli import mnli_loaders
# from dataset.paraphrase import paraphrase_loaders
from dataset.hans_mnli import hans_mnli_loaders

from config import config

if __name__=="__main__":


    task = "hans_mnli"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")

    loaders = hans_mnli_loaders(config=config['tasks'][task], tokenizer=tokenizer)

    print(loaders)

    for domain in loaders:
        print(domain, len(loaders[domain]['train']), len(loaders[domain]['test']), len(loaders[domain]['valid']))


    print("DataLoader Created!")
