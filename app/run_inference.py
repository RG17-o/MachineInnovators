import sys
import os

# Aggiungo la cartella src al path di ricerca dei moduli
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model import load_model_and_tokenizer, infer_sentiment

# Dizionario per mappare gli indici a etichette comprensibili
LABELS = {0: "negativo", 1: "neutro", 2: "positivo"}

def main(text):
    # Carico modello e tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Faccio l'inferenza sul testo passato
    pred_idx = infer_sentiment(text, model, tokenizer)

    # Ottengo l'etichetta corrispondente
    label = LABELS.get(pred_idx, "sconosciuto")

    print(f"Testo: {text}")
    print(f"Sentiment predetto: {label} (indice classe: {pred_idx})")

if __name__ == "__main__":
    # Se l'utente ha inserito del testo come argomento, lo utilizzo
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        # Altrimenti faccio un test con un valore di default
        input_text = "I love this product! It works great."

    main(input_text)