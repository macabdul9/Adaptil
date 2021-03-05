# import torch
from sklearn.metrics import f1_score

def evaluate(model, loader, device):

    true_label = []
    pred_label = []

    model.eval()
    model = model.to(device)

    for batch in loader:

        outputs = model(batch['input_ids'].to(device=device), batch['attention_mask'].to(device=device)).argmax(dim=-1)

        pred_label += outputs.cpu().detach().tolist()
        true_label += batch['label'].cpu().tolist()

    macro_F1 = f1_score(y_true=true_label, y_pred=pred_label, average='macro')

    return macro_F1