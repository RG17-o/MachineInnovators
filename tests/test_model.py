import pytest
from src.model import load_model_and_tokenizer, infer_sentiment

def test_load_model_and_tokenizer():
    """
    Verifico che il modello e il tokenizer vengano caricati correttamente.
    """
    model, tokenizer = load_model_and_tokenizer()
    assert model is not None, "Model non caricato"
    assert tokenizer is not None, "Tokenizer non caricato"

def test_infer_sentiment_valid_output():
    """
    Verifico che infer_sentiment ritorni un indice valido per un testo di esempio.
    """
    model, tokenizer = load_model_and_tokenizer()
    text = "Questo è un ottimo prodotto!"
    pred = infer_sentiment(text, model, tokenizer)
    assert pred in [0, 1, 2], f"Predizione fuori range: {pred}"

def test_infer_sentiment_multiple_texts():
    """
    Verifico infer_sentiment su più testi con sentiment atteso diverso.
    Non confronto etichetta esatta ma assicuriamo output valido.
    """
    model, tokenizer = load_model_and_tokenizer()
    testi = [
        "Mi piace molto questo servizio.",
        "È un giorno come tanti.",
        "Sono deluso dal prodotto."
    ]
    for text in testi:
        pred = infer_sentiment(text, model, tokenizer)
        assert pred in [0, 1, 2], f"Predizione fuori range per testo: {text}"

if __name__ == "__main__":
    pytest.main()