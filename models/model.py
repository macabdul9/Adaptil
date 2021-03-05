import torch.nn as nn
from transformers import AutoModel, AutoConfig

class Model(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.base_model = AutoModel.from_pretrained(configmodel)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=config.dim, out_features=self.config.dim),
            nn.ReLU(),
            nn.Dropout(self.config.seq_classif_dropout),
            nn.Linear(in_features=config.dim, out_features=config.dim),  # change it to config.num_labels
        )

    def forward(self, input_ids, attention_mask=None):

        last_hidden_state = self.base_model(input_ids, attention_mask)[0]
        cls_token = last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits


