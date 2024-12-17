# Progetto-Visione-e-Percezione
Point Cloud classification via Image  Segmentation

Questo progetto utilizza un modello di segmentazione per identificare oggetti specifici in una nuvola di punti 3D. Il modello prende in input un'immagine, esegue la segmentazione e ricostruisce una nuova nuvola di punti in cui le parti segmentate sono evidenziate in colore differente.

## Requisiti
Prima di eseguire il codice, assicurarsi di avere:

- Una GPU compatibile con CUDA, se disponibile, per accelerare l'addestramento del modello.
- Jupyter Notebook o Jupyter Lab per eseguire il notebook `.ipynb`.

## Dipendenze
- Per poter eseguire il codice, è necessario disporre di un ambiente con **Python 3.12.5** installato.
- Le librerie necessarie verranno scaricate eseguendo la prima cella del notebook. Verranno installate le seguenti librerie:
  - `keras`
  - `tensorflow_datasets`
  - `tensorflow`
  - `tensorflow-examples`
  - `numpy` (v. 1.26.0)
  - `pycocotools`
  - `scikit-image`
  - `pyntcloud`
  - `torch` 

## Struttura del Progetto

### Struttura Iniziale
All'inizio, l'unica cartella necessaria è la cartella **Progetto**, contenente i seguenti file di base:
```
├── Progetto/                               # Cartella principale del progetto
|   |
│   ├── codice.ipynb                        # Notebook principale
│   │               
│   └── README.md                           # File README con descrizione del progetto
```

### Struttura Completa
Dopo aver eseguito tutto il progetto, inclusa l'esecuzione e l'addestramento del modello, la struttura della cartella si amplierà come segue:
```
├── Progetto/                               # Cartella principale del progetto
│   ├── COCO dataset/                       # Cartella per il dataset COCO
│   │   ├── images/
│   │   │   ├── train2017/                  # Immagini per il training
│   │   │   ├── val2017/                    # Immagini per la validazione
│   │   │   └── test/                       # Immagini per il test
│   │   └── annotation/
│   │       ├── instances_train2017/        # Annotazioni per il training
│   │       └── instances_val2017/          # Annotazioni per la validazione
│   ├── Modello/                            # Cartella per il modello e le metriche
│   │   ├── modello_vgg.keras               # Modello VGG in formato Keras
│   │   ├── modello_resnet.keras            # Modello ResNet in formato Keras
│   │   ├── history.json                    # File JSON con la cronologia delle metriche
│   │   ├── tabella_metriche_train.png      # Tabella metriche per il training
│   │   └── tabella_metriche_val.png        # Tabella metriche per la validazione
│   ├── output/                             # Cartella per le predizioni salvate
│   │   └── image...                        # File di output delle predizioni
│   ├── codice.ipynb                        # Notebook principale
│   │               
│   └── README.md                           # File README con descrizione del progetto

```

### Contenuto della Cartella
1. **COCO dataset/**: contiene il dataset COCO organizzato in sotto-cartelle per le immagini di training e validazione, nonché le annotazioni per ciascun set.
   - **images/**: Include le immagini utilizzate per l'addestramento e la validazione.
     - **train2017/**: Immagini per il set di addestramento.
     - **val2017/**: Immagini per il set di validazione.
   - **annotation/**: Include le annotazioni di segmentazione per ciascuna immagine.
     - **instances_train2017/**: Annotazioni per il set di addestramento.
     - **instances_val2017/**: Annotazioni per il set di validazione.
   - la cartella di test viene generata dal codice a partire dalla cartella val


2. **Modello/**: contiene il modello addestrato e i file che descrivono le metriche associate per monitorare il progresso dell'addestramento e le prestazioni del modello.
   - `modello.keras`: File del modello addestrato salvato in formato Keras.
   - `history.json`: File JSON che contiene la cronologia delle metriche di addestramento, utile per il tracciamento delle prestazioni (perdita, accuratezza) nel tempo.
   - `tabella_metriche_train.png`: Riepilogo tabellare delle metriche di valutazione per il dataset di training, che facilita un confronto chiaro dei risultati.
   - `tabella_metriche_val.png`: Riepilogo tabellare delle metriche di valutazione per il dataset di validazione, che facilita un confronto chiaro dei risultati.

3. **codice.ipynb**: file notebook contenente l'intero codice suddiviso in celle.

4. **README.md**: documentazione principale che descrive l’obiettivo del progetto, l’organizzazione della cartella, le dipendenze e le istruzioni per l’uso.

# Istruzioni per l'esecuzione del progetto

Per eseguire il progetto correttamente, seguire i passaggi descritti di seguito.

## 1. Configurazione dell'ambiente virtuale

### Opzione 1: Utilizzo di Anaconda
   - Installare **Anaconda** se non è già presente sul proprio computer. Anaconda include Python e offre strumenti per la gestione degli ambienti virtuali.
   - Creare un nuovo ambiente virtuale con la versione **Python 3.12.5**. Per fare ciò:
     1. Aprire **Anaconda Navigator**.
     2. Selezionare **Environments** dal menu a sinistra.
     3. Cliccare su **Create** in basso.
     4. Dare un nome all'ambiente (ad esempio, `ambiente_progetto`) e selezionare la versione di **Python 3.12.5**.
     5. Cliccare su **Create** per creare l'ambiente.

### Opzione 2: Creazione di un ambiente virtuale senza Anaconda
   - Se non si desidera utilizzare Anaconda, è possibile creare un ambiente virtuale utilizzando Python direttamente:
     1. Aprire il terminale.
     2. Eseguire il comando per creare un ambiente virtuale:
        ```bash
        python3 -m venv nome_tuo_ambiente
        ```
     3. Attivare l’ambiente virtuale:
        - **Linux/macOS**: `source nome_tuo_ambiente/bin/activate`
        - **Windows**: `.\nome_tuo_ambiente\Scripts\activate`

## 2. Installazione di Jupyter Notebook

   ### Con Anaconda Navigator
   - In **Anaconda Navigator**, è possibile installare Jupyter Notebook tramite l’interfaccia grafica:
     1. Nella sezione **Environments**, selezionare l’ambiente appena creato (`ambiente_progetto`).
     2. Nella lista dei pacchetti (Packages), assicurarsi che sia selezionata la voce **Not Installed** per visualizzare solo i pacchetti non installati.
     3. Cercare **Jupyter** nella barra di ricerca.
     4. Selezionare **Jupyter Notebook** dalla lista dei risultati e cliccare su **Apply** per installarlo.

   ### Senza Anaconda
   - Se non si usa Anaconda, installare Jupyter Notebook tramite pip:
     ```bash
     pip install jupyter
     ```

## 3. Configurazione del percorso di progetto
   - Aprire il file **`codice.ipynb`** in **Jupyter Notebook**.
   - Andare alla **cella 3** del notebook.
   - Sostituire il contenuto della variabile `project_dir` con il percorso completo della cartella principale del progetto sul proprio computer. Ad esempio:
     ```python
     project_dir = "C:/user/mariorossi/Progetto"
     ```

## 4. Addestramento del modello
   - È possibile scegliere se utilizzare il modello già addestrato o eseguirne un nuovo addestramento.
   - Nella **cella 3**, impostare la variabile `addestramento` come `True` per addestrare il modello o `False` per usare il modello già addestrato:
     ```python
     addestramento = True  # Cambia in False per usare il modello pre-addestrato
     ```
## 5. Esecuzione del codice
   - Dopo aver configurato le variabili, eseguire tutte le celle del notebook selezionando **"Run all cells"** dal menu di Jupyter Notebook o premendo `Shift + Enter` su ciascuna cella.

Seguendo queste istruzioni, sarà possibile configurare e utilizzare il progetto correttamente.

## Predizione su un'Immagine Personalizzata

Alla fine del file **`codice.ipynb`** è presente una cella che consente di effettuare una **predizione su un'immagine personalizzata** caricata dal proprio computer. Questa funzionalità permette di analizzare un'immagine a scelta e di ottenere i seguenti risultati:

1. **Immagine Originale**: l'immagine caricata dall'utente.
2. **Maschera Segmentata**: la maschera generata dal modello, che evidenzia le aree segmentate dell'immagine.
3. **Depth Map**: una mappa di profondità stimata, utile per rappresentare la distanza di ciascun punto dall'osservatore.
4. **Nuvola di Punti Originale**: la ricostruzione 3D della scena basata sull'immagine originale.
5. **Nuvola di Punti Segmentata**: la nuvola di punti, dove le aree segmentate sono evidenziate con colori differenti.

### Come Utilizzare la Funzione di Predizione

1. **Caricamento dell'Immagine**:
   - Alla fine del notebook, troverai una cella che ti permetterà di selezionare un file immagine direttamente dal tuo computer.

2. **Esecuzione della Predizione**:
   - Dopo aver selezionato l'immagine, il modello eseguirà la segmentazione e mostrerà le diverse visualizzazioni.

3. **Visualizzazione dei Risultati**:
   - I risultati saranno mostrati in una griglia con:
     - L'immagine originale.
     - La maschera segmentata (sovrapposta).
     - La depth map.
     - Le nuvole di punti, visualizzate interattivamente.

### Requisiti per le Immagini
- Le immagini caricate devono essere in un formato supportato, come **JPG**, **PNG**, o **JPEG**.
- La risoluzione delle immagini non dovrebbe essere troppo alta per evitare rallentamenti nell'elaborazione.

Questa funzione ti permette di testare il modello su immagini personalizzate, offrendo una visualizzazione completa dei risultati della segmentazione e delle relative applicazioni in 3D.
