from model import load_model_and_tokenizer, infer_sentiment

def main():
    model, tokenizer = load_model_and_tokenizer()

    test_text = "I love using this product! It's fantastic and works great."
    prediction = infer_sentiment(test_text, model, tokenizer)
    print(f"Testo: {test_text}")
    print(f"Sentiment predetto (indice classe): {prediction}")

if __name__ == "__main__":
    main()