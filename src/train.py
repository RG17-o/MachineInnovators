import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score
import numpy as np 

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def compute_metrics(eval_pred):
    # eval_pred è una tupla con predictions e labels
    predictions, labels = eval_pred 
    
    # Prendo l'indice della classe con il punteggio più alto (argmax)
    predictions = np.argmax(predictions, axis=1)

    # Calcolo l'accuratezza
    accuracy = accuracy_score(labels, predictions)
    
    # Il Trainer si aspetta un dizionario
    return {"accuracy": accuracy}


def main():
    # Carico tokenizer e modello pre-addestrato
    # Aggiungo argomento per numero di campioni
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max_samples", type=int, default=0, help="Max number of samples to use for smoke test (0=full dataset).") # NUOVO ARGOMENTO
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # Carico dataset di esempio (SST2)
    dataset = load_dataset("glue", "sst2")

    # Limito il datasetper smoke test

    if args.max_samples > 0:
        dataset["train"] = dataset["train"].select(range(min(args.max_samples, len(dataset["train"]))))
        # Limita anche validation e test per sicurezza
        dataset["validation"] = dataset["validation"].select(range(min(100, len(dataset["validation"]))))
        dataset["test"] = dataset["test"].select(range(min(100, len(dataset["test"]))))
        print(f"ATTENZIONE: Limitato dataset per smoke test a {len(dataset['train'])} campioni.")

    # Funzione di tokenizzazione
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding=True, max_length=128)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Settaggio di feature e label
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Rimuovo le colonne originali non necessarie al Trainer
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])

    # Creo il Data Collator (necessario per il padding dinamico nel batch)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Parametri di training
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs= args.num_train_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
    )

    # Trainer per gestire training e valutazione
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        )

    # Avvio il training
    trainer.train()

    # Salvo il modello addestrato
    trainer.save_model("./trained_model")

if __name__ == "__main__":
    main()