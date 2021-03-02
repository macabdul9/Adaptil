import torch.nn as nn
from transformers import AutoModel, AutoConfig

class Model(nn.Module):
    
    def __init__(self, model_name, num_classes=2): 
        
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name)
        
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.config.dim, out_features=self.config.dim),
            nn.ReLU(),
            nn.Dropout(self.config.seq_classif_dropout),
            nn.Linear(in_features=self.config.dim, out_features=num_classes),
        )

    def forward(self, input_ids, attention_mask=None):
        
        last_hidden_state, _ = self.base_model(input_ids, attention_mask)[0] 
        
        cls_token = last_hidden_state[:, 0] 
        
        logits = self.classifier(cls_token)
        
        return logits
        
        
        