# import torch
from sklearn.metrics import classification_report

def evaluate(model, loader, device):
    
    true_label = []
    pred_label = []
    
    model.eval()
    model = model.to(device)
    
    for batch in loader:
        
        outputs = model(batch['input_ids'], batch['attention_mask']).argmax(dim=-1)
        
        true_label += outputs.cpu().detach().tolist()
        pred_label += batch['label'].cpu().tolist()
    
    report = classification_report(y_true=true_label, y_pred=pred_label)
    return true_label, pred_label, report 
    