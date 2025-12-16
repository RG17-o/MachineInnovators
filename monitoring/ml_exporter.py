import time
import random
from prometheus_client import start_http_server, Gauge

# 1. Definisco la Metrica
# Gauge è un valore che può salire e scendere liberamente (es. Accuratezza)
ACCURACY_GAUGE = Gauge('ml_model_accuracy', 'Accuratezza del modello ML')

# Funzione per simulare la raccolta della metrica di accuratezza
def collect_metrics():
    """Simulo l'ottenimento di una metrica di accuratezza fittizia."""
    
    # QUI DOVRAI INTEGRARE IL TUO VERO CODICE ML
    # Ad esempio: accuracy = train_and_get_accuracy()
    
    # Per ora, simuliamo un valore casuale tra 0.85 e 0.95
    accuracy = random.uniform(0.85, 0.95)
    
    # Imposto il valore nel Gauge di Prometheus
    ACCURACY_GAUGE.set(accuracy)
    
    print(f"Metrica aggiornata: Accuratezza = {accuracy:.4f}")

if __name__ == '__main__':
    # Avvio il server HTTP per esporre le metriche sulla porta 8000
    start_http_server(8000)
    print("Exporter avviato sulla porta 8000.")
    
    # Loop continuo per aggiornare le metriche (una volta ogni 10 secondi)
    while True:
        collect_metrics()
        time.sleep(10)