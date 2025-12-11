from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import load_dataset
import torch

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def main():
    # Carico tokenizer e modello pre-addestrato
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # Carico dataset di esempio (SST2)
    dataset = load_dataset("glue", "sst2")

    # Funzione di tokenizzazione
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding=True, max_length=128)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Settaggio di feature e label
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Creo il Data Collator (necessario per il padding dinamico nel batch)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Parametri di training
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
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
    )

    # Avvio il training
    trainer.train()

    # Salvo il modello addestrato
    trainer.save_model("./trained_model")

if __name__ == "__main__":
    main()