# MachineInnovators
**- Visione del Progetto**

MachineInnovators Inc. sviluppa soluzioni di Machine Learning pronte per la produzione. Questo progetto affronta la sfida della gestione della reputazione online automatizzando l'analisi del sentiment sui social media. L'obiettivo non è solo classificare il testo, ma garantire che il sistema sia affidabile, testato e monitorato costantemente attraverso metodologie MLOps.


**- Architettura e Scelte Progettuali**

 *-1. Il Modello: Da FastText a RoBERTa*
Sebbene la traccia iniziale prevedesse l'utilizzo FastText , ho scelto di implementare il modello multilingua Twitter-RoBERTa-base (cardiffnlp/twitter-roberta-base-sentiment-latest).
Il motivo di questa scelta da parte mia risiede nel fatto che RoBERTa utilizza l'architettura Transformer, in grado di comprendere il contesto e le sfumature del linguaggio (ad esempio l'ironia) meglio rispetto ai modelli basati su word-embeddings statici come FastText. Essendo pre-addestrato dettagliatamente su contenuti derivanti da Twitter, è ideale per essere applicato al contesto dei social media.

Performance: Il modello ha raggiunto un'accuratezza del 77.15% sul validation set (tweet_eval).

 *-2. Pipeline CI/CD (GitHub Actions)*
Ho automatizzato il ciclo di vita del software:

Integrazione Continua (CI): Ad ogni push, vengono eseguiti test automatici con pytest per verificare il caricamento del modello e la correttezza delle inferenze.

Distribuzione Continua (CD): Se i test hanno risultato positivo sul branch main, il modello viene caricato in modo automatico sull'Hugging Face Hub. In questo modo riusciamo a garantire che la versione in produzione sia sempre l'ultima validata.


*-3. Monitoraggio Real-Time (Prometheus & Grafana)*
In un contesto produttivo reale all'interno di una azienda, è fondamentale sapere se il modello vede delle performance non più idonee.

Custom Exporter: Ho sviluppato un servizio Python che valuta periodicamente il modello e ne espone le metriche.

Visualizzazione: Ho scelto d utilizzare Grafana per monitorare l'accuratezza in tempo reale. Se l'accuratezza scende sotto una certo valore di soglia, l'azienda può intervenire mediante un retraining.



**- Struttura del Repository**

src/: Codice sorgente per caricamento, training e inferenza del modello.

app/: Script pronti all'uso per l'inferenza rapida.

monitoring/: Configurazione Docker Compose per lo stack Prometheus e Grafana e codice dell'exporter.

tests/: Suite di test per garantire la stabilità del codice.

.github/workflows/: Definizione della pipeline CI/CD.

**- Come Iniziare**

*-1. Inferenza Locale*
Per testare rapidamente il modello è possibile digitare direttamente nel terminale:

python app/run_inference.py "Questo progetto mi è piaciuto tantissimo!"

*-2. Avvio del Monitoraggio*
Per lanciare l'intero stack di monitoraggio con Docker si può digitare nel terminale :

cd monitoring
docker-compose up -d


Fatto ciò, le metriche saranno visibili su Grafana all'indirizzo localhost:3000.

**- Risultati Ottenuti**

Il sistema è in grado di processare flussi di dati social e fornire un feedback immediato sulla reputazione aziendale. Grazie al monitoraggio continuo, MachineInnovators garantisce che l'accuratezza rimanga stabile nel tempo, permettendo risposte tempestive ai cambiamenti di sentiment degli utenti.

Riporto una immagine, relativa all'accuracy, proveniente dalla dashboard creata su Grafana:

![Dashboard Accuratezza Grafana](grafana-accuracy.png)
