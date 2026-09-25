# CardioExplain RAG

Dieses Projekt ist ein Prototyp für die Masterarbeit:

**Implementierung und Evaluierung eines KI-gestützten RAG-Systems zur laienverständlichen Erklärung kardiologischer Fachbegriffe in deutschsprachigen Arztbriefen auf Basis der README-Datenbank**

## Idee

Das System nutzt eine kardiologisch gefilterte README-Teildatenbank als Wissensbasis.  
Aus der TSV-Datei wird ein FAISS-Vektorindex erstellt. Danach kann eine Streamlit-App deutschsprachige Arztbrief-Auszüge und Nutzerfragen verarbeiten.

## Projektstruktur

```text
cardio_rag_masterarbeit/
├── app.py
├── build_index.py
├── rag_pipeline.py
├── config.py
├── requirements.txt
├── .env.example
├── data/
│   └── readme_kardiologie.tsv
├── storage/
└── evaluation/
    ├── eval_questions.csv
    └── eval_results.csv
```

## Installation in Visual Studio Code

### 1. Virtuelle Umgebung erstellen

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Pakete installieren

```bash
pip install -r requirements.txt
```

### 3. Ollama installieren und Modell starten

Ollama muss lokal installiert sein. Danach z. B.:

```bash
ollama pull llama3.1:8b
ollama serve
```

In einer zweiten Konsole dann die App starten.

### 4. Umgebungsvariablen setzen

Kopiere `.env.example` zu `.env`.

```bash
cp .env.example .env
```

Unter Windows kannst du die Datei einfach in VS Code kopieren/umbenennen.

### 5. Index bauen

```bash
python build_index.py
```

### 6. App starten

```bash
streamlit run app.py
```

## Evaluation

Eine einfache Retrieval-Evaluation ist vorbereitet:

```bash
python evaluate_retrieval.py
```

Dabei wird geprüft, ob erwartete Fachbegriffe unter den Top-k-Retrieval-Treffern erscheinen.

## Hinweis

Das System ist ein Forschungsprototyp. Es ersetzt keine medizinische Beratung.
