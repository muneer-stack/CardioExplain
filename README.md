# CardioExplain RAG

CardioExplain ist ein Forschungsprototyp für die Masterarbeit:

**Implementierung und Evaluierung eines KI-gestützten RAG-Systems zur laienverständlichen Erklärung kardiologischer Fachbegriffe in deutschsprachigen Arztbriefen auf Basis der README-Datenbank**

Die Anwendung verbindet eine deutschsprachige, kardiologisch gefilterte README-Wissensbasis mit einer Suche nach passenden Einträgen und einem über Ollama eingebundenen Sprachmodell. Nutzer können einen Arztbrief als PDF oder TXT hochladen oder Text einfügen und Fragen zu den darin enthaltenen Fachbegriffen stellen.

## Dokumentierter Stand

Diese Beschreibung dokumentiert die im Repository enthaltenen Dateien `rag_pipeline.py`, `app.py`, `build_index.py` und `config.py`. Die Pipeline umfasst eine regelbasierte Zielbegriffserkennung, eine technische Abdeckungsprüfung und eine auf die erfragten Begriffe ausgerichtete Auswahl des Arztbriefkontexts.

## Funktionen

- PDF- und TXT-Upload sowie direkte Texteingabe.
- Gesonderte Anzeige wörtlich erkannter deutscher und englischer Datenbankbegriffe im Arztbrief einschließlich der zugehörigen Datensatz-IDs.
- Kombination aus exakter Begriffssuche und semantischer Suche mit FAISS.
- Regelbasierte Bestimmung der erfragten Zielbegriffe und Suche nach enthaltenen Datenbankformen, wenn eine exakte Form fehlt.
- Auswahl eines lokalen Arztbriefausschnitts zur Einordnung der Frage.
- Deutsche Begriffserklärungen mit `llama3.1:8b` über Ollama als Standardmodell.
- Anzeige der ausgewählten README-Treffer, des lokalen Kontexts und der Retrieval-Anfrage.

Die PDF-Verarbeitung verwendet `pypdf` zur Textextraktion. Eine OCR-Erkennung für reine Bildscans ist nicht implementiert.

## Projektdateien

| Datei oder Pfad | Aufgabe |
| --- | --- |
| `app.py` | Streamlit-Oberfläche, Dokumentimport und Anzeige der Antworten und Treffer. |
| `rag_pipeline.py` | Begriffserkennung, Kontextauswahl, Retrieval, Abdeckungsprüfung und Antwortgenerierung. |
| `build_index.py` | Aufbau des FAISS-Indexes, Term-Katalogs und Index-Manifests. |
| `config.py` | Datenpfade, Modellkonfiguration und Standardparameter. |
| `requirements.txt` | Python-Abhängigkeiten. |
| `data/readme_kardiologie_bilingual_audit.tsv` | Vorbereitete bilinguale kardiologische Wissensbasis. |
| `storage/faiss_cardio_readme/` | Indexdateien `index.faiss`, `index.pkl` und `index_manifest.json`. |
| `storage/term_catalog.csv` | Beim Indexaufbau erzeugter Katalog der Begriffsformen und Metadaten. |

Die Wissensbasis muss am konfigurierten Datenpfad vorhanden sein. Der Index und der Term-Katalog können mit `build_index.py` erzeugt werden.

## Installation und Start

Alle folgenden Befehle werden im Projektordner ausgeführt. Voraussetzung sind Python, die Projektdateien, die vorbereitete TSV-Wissensbasis und eine lokale Ollama-Installation.

### 1. Virtuelle Umgebung erstellen und aktivieren

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Abhängigkeiten installieren

```bash
python -m pip install -r requirements.txt
```

### 3. Ollama-Modell bereitstellen

```bash
ollama pull llama3.1:8b
```

Ollama muss für die Antwortgenerierung erreichbar sein. Falls noch kein Ollama-Dienst läuft, kann er in einer separaten Konsole gestartet werden:

```bash
ollama serve
```

### 4. Konfiguration prüfen

Die geprüfte `config.py` verwendet folgende Standardwerte:

| Parameter | Standardwert |
| --- | --- |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| `OLLAMA_MODEL` | `llama3.1:8b` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `DEFAULT_TOP_K` | `5` |
| `MAX_PATIENT_TEXT_CHARS` | `6000` |

Die drei Modell- und Verbindungsvariablen können über Umgebungsvariablen oder eine optional im Projektordner angelegte `.env`-Datei gesetzt werden:

```dotenv
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

`DEFAULT_TOP_K`, `MAX_PATIENT_TEXT_CHARS` und die Datenpfade werden in der geprüften Fassung direkt in `config.py` festgelegt. Das Embedding-Modell wird auf der CPU ausgeführt; beim ersten Laden können Modelldateien heruntergeladen werden.

### 5. Index aufbauen

```bash
python build_index.py
```

Die TSV-Datei benötigt die Spalten `record_id`, `mention_count`, `ann_text_en`, `ann_text_de`, `general_definition_en`, `lay_definition_en` und `lay_definition_de`. Das Skript erstellt pro vorbereitetem Datensatz genau ein Retrieval-Dokument. Unterschiedliche Einträge zu derselben Begriffsform bleiben getrennt erhalten. Doppelte Datensatz-IDs führen zu einem Abbruch.

Beim Laden prüft die Pipeline unter anderem den Datenbank-Hash, das Embedding-Modell und die Dokumentanzahl gegen das Index-Manifest. Nach Änderungen an der Wissensbasis oder am Embedding-Modell muss der Index neu erstellt werden. Änderungen ausschließlich am Prompt oder an der Suchlogik erfordern keinen neuen Indexaufbau. Nach Änderungen am Code oder an der Konfiguration die Anwendung neu starten, damit zwischengespeicherte Komponenten erneut geladen werden.

### 6. Anwendung starten

```bash
python -m streamlit run app.py
```

Anschließend einen Arztbrief hochladen oder Text einfügen und eine Frage stellen, beispielsweise: „Was bedeutet PTCA mit DES in der LAD in diesem Arztbrief?“

## Verarbeitung einer Frage

### Zielbegriffe und Begriffsformen

Die Pipeline gleicht die Frage mit den deutschen und englischen Begriffsformen der Wissensbasis ab. Zusätzlich berücksichtigt sie den Text nach Formulierungen wie „Was bedeutet“ oder „Wofür steht“ und auffällige Schreibweisen wie Großbuchstaben-Abkürzungen oder Ausdrücke mit Bindestrichen.

Für einen Zielbegriff wird zuerst nach einer normalisierten exakten Datenbankform gesucht. Fehlt diese, prüft der Code, ob eine längere bekannte Datenbankform in der bereinigten Zeichenfolge enthalten ist. So kann `hs-cTroponin` dem Datenbankbegriff `Troponin` zugeordnet werden, wenn dieser vorhanden ist. Diese Zeichenprüfung ist keine medizinische Bedeutungsprüfung und kann auch unpassende Zuordnungen erzeugen.

Der wörtliche Abgleich im gesamten Arztbrief ist eine gesonderte Funktion. Er gibt pro normalisierter Begriffsform das erste gefundene Vorkommen zurück und zählt nicht sämtliche Erwähnungen. Reine Zahlen sowie sehr kurze Formen, die nicht vollständig großgeschrieben sind, werden ausgelassen. Die Suchmuster berücksichtigen Wortgrenzen sowie Varianten von Leerzeichen und Bindestrichen.

### Arztbriefkontext

Zunächst sucht die Pipeline im Arztbrief nach längeren Wortfolgen aus der Frage mit vier bis achtzehn Wörtern. Bei vorhandenen Zielbegriffen müssen die geprüften Wortfolgen einen formalen Bezug zu diesen haben. Ohne passende Wortfolge sucht sie nacheinander nach Zielbegriffen und weiteren erkannten Formen.

Um die Fundstelle werden zunächst bis zu 700 Zeichen davor und danach ausgewählt. Umliegende Zeilenumbrüche können den Ausschnitt erweitern. Ohne passende Fundstelle dienen die ersten 1.500 Zeichen als Ersatzkontext.

### Auswahl der README-Treffer

Die Retrieval-Anfrage enthält die Nutzerfrage, die Zielbegriffe, gegebenenfalls zugeordnete Datenbankformen und den lokalen Arztbriefkontext. Für die semantische Suche werden `max(50, top_k * 10)` Kandidaten angefordert.

Für jeden Zielbegriff mit zugeordneten Datenbankformen wird zunächst ein noch nicht ausgewählter exakter Datensatz berücksichtigt, solange Plätze verfügbar sind. Bei mehreren Kandidaten dient der semantische Rang zur Sortierung; bei gleichem Rang wird die höhere ursprüngliche Erwähnungshäufigkeit bevorzugt. Verbleibende Plätze werden mit semantischen Treffern aufgefüllt. Die Auswahl enthält höchstens `top_k` unterschiedliche Datensätze, standardmäßig fünf.

### Abdeckungsprüfung und Antwort

Die technische Abdeckungsprüfung unterscheidet drei Zustände:

| Zustand | Bedeutung |
| --- | --- |
| `exact` | Eine normalisierte exakte Datenbankform wurde gefunden. |
| `derived` | Die genaue Form fehlt; eine darin enthaltene längere Datenbankform wurde zugeordnet. |
| `unresolved` | Keine exakte oder durch diese Regel zuordenbare Form wurde gefunden. |

Zusätzlich wird festgehalten, welche zugehörigen Datensätze in die ausgewählten Treffer übernommen wurden. Diese Informationen werden dem Modell im Prompt bereitgestellt. Sie prüfen die formale Abdeckung, nicht die medizinische Richtigkeit oder die kontextbezogene Eignung der Treffer.

Bei Arztbriefen mit höchstens `MAX_PATIENT_TEXT_CHARS` Zeichen erhält das Modell den vollständigen verfügbaren Text und gegebenenfalls zusätzlich den lokalen Ausschnitt. Bei längeren Texten wird bevorzugt der lokale Ausschnitt bis zur Zeichengrenze übergeben, andernfalls der Anfang des Arztbriefs. Die Grenze betrifft diese Auswahl des Arztbrieftexts und nicht die Gesamtlänge des Prompts einschließlich Frage, Wissenseinträgen und Abdeckungsprüfung.

Der Prompt verlangt eine kurze deutsche Erklärung auf ungefähr dem Niveau der siebten bis achten Klasse, eine Einordnung in den Arztbrief und bei Bedarf einen Hinweis auf fehlende oder uneindeutige Informationen. Er untersagt unbelegte medizinische Ergänzungen, erfundene Abkürzungsauflösungen sowie individuelle Diagnose- oder Therapieempfehlungen. Diese Vorgaben werden nicht durch einen automatischen medizinischen Faktencheck abgesichert. Die Temperatur des Sprachmodells ist auf `0.0` gesetzt.

Die überarbeitete Pipeline gibt zusätzlich `focus_terms` und `retrieval_coverage` im Ergebnisobjekt zurück. Die geprüfte `app.py` zeigt diese Felder noch nicht gesondert an. Sichtbar sind die ausgewählten Treffer, die in der Frage erkannten Datenbankbegriffe, der lokale Kontext und die Retrieval-Anfrage. Frühere Chatnachrichten werden angezeigt, aber nicht als Gesprächskontext an die Antwortfunktion übergeben.

## Evaluation und Grenzen

Die Untersuchung in der Masterarbeit betrachtet die dokumentbezogene Begriffserkennung und das Retrieval, die Zuordnung im Kontext beziehungsweise Disambiguierung sowie die fachliche und sprachliche Qualität der Antworten. Eine gefundene Begriffsform ist dabei getrennt von einer passenden medizinischen Erklärung zu bewerten.

Fehlerhafte Wissenseinträge, Übersetzungsprobleme, fehlende Begriffsformen und mehrdeutige Abkürzungen können zu unpassenden Treffern oder falschen Antworten führen. Die Anzeige abgerufener Einträge ermöglicht keine vollständige Quellenzuordnung auf Satzebene. Die Begriffs- und Kontextauswahl beruht auf Regeln und Zeichenübereinstimmungen; sie ersetzt keine medizinische Analyse.

CardioExplain ist ein Forschungsprototyp und ersetzt keine medizinische Beratung.
