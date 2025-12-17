# Uso un'immagine Python leggera
FROM python:3.11-slim

# Imposto la directory di lavoro
WORKDIR /app

# Copio i files di dipendenza e li installo
COPY monitoring/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copio il codice dell'exporter
# Copio il modulo principale (ml_exporter.py) e la cartella dei sorgenti (src/)
COPY monitoring/ml_exporter.py .
COPY src/ src/  

# Aggiungo /app al PYTHONPATH in modo che gli script possano trovare i moduli in src/ ed imposto la variabile d'ambiente
ENV PYTHONPATH /app:${PYTHONPATH}

# Eseguo l'exporter quando il container parte
CMD ["python", "ml_exporter.py"]