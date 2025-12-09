from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def load_model_and_tokenizer():
    """
    Carica e restituisce il modello pre-addestrato e il tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return model, tokenizer

def infer_sentiment(text, model, tokenizer, device=None):
    """
    Esegue inferenza di sentiment su una singola frase di testo.
    Restituisce l’indice della classe predetta (0=negativo, 1=neutro, 2=positivo).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    return pred