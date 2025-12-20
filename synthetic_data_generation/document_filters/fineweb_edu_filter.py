import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pdb


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def fineweb_quality_filter(passages):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
    model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier").to(device)
    
    if isinstance(passages, str):
        passages = [passages]

    scores = []
    for text in passages:
        inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1).float().detach().cpu().numpy()
        score = logits.item()
        result = {
            "text": text,
            "score": score,
            "int_score": int(round(max(0, min(score, 5)))),
        }
        scores.append(score)
    
    return scores
