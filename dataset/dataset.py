import os 
from tqdm import tqdm
from .SADataset import SADataset
from torch.utils.data import DataLoader, random_split

# Statement from Abdul: I intend to write a generic (task and data agnostic) dataset class but BHavitvya told me go through with this approach

# sa data loader
def create_sa_loaders(tokenizer, root_dir, domains =["books", "dvd", "electronics", "kitchen_housewares"], batch_size=8):
    "root dir will have all the domain files with domain name"
    
    loaders = {}
    files = os.listdir(root_dir)
    
    for domain in tqdm(domains):
        
        d = os.path.join(root_dir, domain+".csv")
        
        # create the dataset for current domain
        dataset = SADataset(tokenizer=tokenizer, file_name=d)
        
        
        train_dataset, valid_dataset = random_split(dataset=dataset, lengths=[int(len(dataset)*0.80), len(dataset) - int(len(dataset)*0.20)])
        
        
        # create loader
        train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(dataset = valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        
        # put the domain laoder into dictionary
        loaders[domain] = {
            "train":train_loader,
            "valid":valid_loader
        }
    
    return loaders


def create_mlni_loaders(tokenizer, batch_size=8):
    pass
        
    