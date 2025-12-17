import time
import torch
import numpy as np
from prometheus_client import start_http_server, Gauge
from datasets import load_dataset 
from torch.utils.data import DataLoader

# Importa la funzione dal tuo modulo model.py (richiede che src/ sia copiato nel container)
from src.model import load_model_and_tokenizer 
# NOTA: Assicurati che il tuo model.py sia aggiornato con le funzioni necessarie
# e che il Dockerfile copi src/ e imposti il PYTHONPATH.

# --- CONFIGURAZIONE E CARICAMENTO ---
# Carica il modello e il tokenizer solo una volta all'avvio
print("Caricamento modello e tokenizer...")
# La funzione load_model_and_tokenizer gestisce l'importazione da Hugging Face
model, tokenizer = load_model_and_tokenizer() 

# Configura il device (CPU o CUDA, se disponibile)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

# Carica e pre-processa il dataset di validazione solo una volta all'avvio
print("Caricamento e tokenizzazione dataset di validazione...")
dataset = load_dataset("tweet_eval", "sentiment")

# Funzione di tokenizzazione (copiata dal tuo notebook)
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_validation_set = dataset["validation"].map(preprocess_function, batched=True)
tokenized_validation_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Creiamo il DataLoader per iterare i dati
val_loader = DataLoader(tokenized_validation_set, batch_size=16) 


# --- LOGICA DI PROMETHEUS ---
# 1. Definisco la Metrica
ACCURACY_GAUGE = Gauge('ml_model_accuracy', 'Accuratezza del modello ML sul set di validazione')


# --- FUNZIONE DI VALUTAZIONE ---
def calculate_validation_accuracy():
    """Valuta il modello sul set di validazione e restituisce l'accuratezza."""
    
    all_preds = []
    all_labels = []
    
    model.eval() # Imposta il modello in modalità valutazione
    with torch.no_grad():
        for batch in val_loader:
            # Sposta i dati sul device
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            # Esegui l'inferenza
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Calcola le predizioni e sposta su CPU per l'aggregazione
            preds = torch.argmax(logits, dim=1).cpu() 
            labels = batch["label"].cpu()            
            
            all_preds.append(preds)
            all_labels.append(labels)
            
    # Concatenazione di tutte le predizioni e label
    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)
    
    # Calcolo accuracy
    accuracy = (all_preds_tensor == all_labels_tensor).sum().item() / len(all_labels_tensor)
    return accuracy

def collect_metrics():
    """Calcola l'accuratezza e aggiorna il Gauge di Prometheus."""
    
    # Esegui il calcolo costoso dell'accuratezza
    accuracy = calculate_validation_accuracy()
    
    # Imposta il valore nel Gauge di Prometheus
    ACCURACY_GAUGE.set(accuracy)
    
    print(f"Metrica aggiornata: Accuratezza sul Validation Set = {accuracy:.4f}")

if __name__ == '__main__':
    # Avvia il server HTTP per esporre le metriche sulla porta 8000
    start_http_server(8000)
    print("Exporter avviato sulla porta 8000.")
    
    # Loop continuo per aggiornare le metriche (una volta ogni 5 minuti)
    while True:
        collect_metrics()
        # Tempo di attesa in secondi (5 minuti = 300s)
        time.sleep(30)