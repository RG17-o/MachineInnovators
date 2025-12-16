# Uso un'immagine Python leggera
FROM python:3.11-slim

# Imposto la directory di lavoro
WORKDIR /app

# Copio i files di dipendenza e li installo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copio il codice dell'exporter
COPY ml_exporter.py .

# Eseguo l'exporter
CMD ["python", "ml_exporter.py"]