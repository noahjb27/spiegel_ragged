# Der Spiegel RAG System (1948-1979)

Ein Retrieval Augmented Generation (RAG) System zur Analyse und Durchsuchung des Spiegel-Archivs der Jahrgänge 1948-1979.

## Übersicht

Dieses System ermöglicht die semantische Suche, Analyse und KI-gestützte Auswertung von Artikeln aus dem Spiegel-Archiv. Es kombiniert Vektordatenbanktechnologie mit großen Sprachmodellen, um relevante Textpassagen zu finden und basierend auf diesen Informationen qualitativ hochwertige Antworten zu generieren.

### Hauptfunktionen

- **Semantische Suche**: Finde relevante Inhalte basierend auf Bedeutung, nicht nur auf Schlagwörtern
- **Zeitfenster-Suche**: Analysiere Inhalte über verschiedene Zeitperioden hinweg
- **Schlagwort-Filterung**: Verwende boolesche Ausdrücke (AND, OR, NOT) für präzise Filterung
- **Semantische Erweiterung**: Automatische Erweiterung von Suchbegriffen mit ähnlichen Wörtern
- **KI-gestützte Analyse**: Generiere Zusammenfassungen und Analysen basierend auf gefundenen Texten
- **Mehrsprachige Modelle**: Unterstützung für lokale (HU-LLM) und externe (OpenAI) Sprachmodelle

## Systemanforderungen

- Python 3.8 oder höher
- Internetzugang über HU-Eduroam oder VPN für die Verbindung zum ChromaDB und Ollama Embedding Service
- Optional: OpenAI API-Schlüssel für GPT-4o oder GPT-3.5 Turbo

## Installation

### 1. Repository klonen

```bash
git clone [repository-url]
cd spiegel_ragged
```

### 2. Virtuelle Umgebung erstellen und aktivieren

```bash
python -m venv .venv
source .venv/bin/activate  # Unter Windows: .venv\Scripts\activate
```

### 3. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 4. FastText Word Embeddings herunterladen

Die FastText-Worteinbettungen müssen separat heruntergeladen werden, da sie zu groß für das Repository sind:

1. Laden Sie die Einbettungen von der angegebenen URL herunter: <https://box.hu-berlin.de/f/34c17bfd74b84454b276/?dl=1>
2. Entpacken Sie die Dateien in den `models/` Ordner

**Hinweis zu verfügbaren ChromaDB-Kollektionen:**
Das System ist so konfiguriert, dass es nur die folgenden zwei existierenden ChromaDB-Kollektionen verwendet:

1. `recursive_chunks_3000_300_TH_cosine_nomic-embed-text`
2. `recursive_chunks_2000_400_TH_cosine_nomic-embed-text`

## Ausführen der Anwendung

### Starten der Benutzeroberfläche

```bash
python src/ui/app.py
```

Die Anwendung ist dann unter <http://localhost:7860> verfügbar.

## Komponententests

Testen Sie die Kernkomponenten mit dem Diagnose-Tool:

```bash
python src/utils/component_test.py
```

## Architektur

Die Anwendung besteht aus folgenden Hauptkomponenten:

### Kern-Module

- **RAG Engine** (`src/core/rag_engine.py`): Hauptmodul, das die Suche und Antworterstellung koordiniert
- **Vector Store** (`src/core/vectore_store.py`): Interface zur externen ChromaDB-Vektordatenbank
- **LLM Service** (`src/core/llm_service.py`): Interface zu verschiedenen Sprachmodellen (HU-LLM oder OpenAI)
- **Embedding Service** (`src/core/embedding_service.py`): Lokaler Worteinbettungs-Service für semantische Ähnlichkeitssuche mit FastText

Der Datenfluss in der Anwendung:

1. Der Benutzer gibt eine Suchanfrage und Frage in der UI ein
2. Die RAG Engine sendet eine Anfrage an den externen ChromaDB-Service, um relevante Textabschnitte zu finden
3. Der externe Ollama-Service liefert die Sentence Embeddings für die Ähnlichkeitssuche
4. Der lokale Embedding Service erweitert Schlagwörter mit semantisch ähnlichen Begriffen
5. Die gefundenen Texte werden an das ausgewählte Sprachmodell (HU-LLM oder OpenAI) gesendet
6. Das Sprachmodell generiert eine Antwort, die in der UI angezeigt wird

### Konfiguration

- **Einstellungen** (`src/config/settings.py`): Zentrale Konfigurationsdatei

## Nutzung

### Durchsuchen des Archivs

1. Geben Sie eine Suchanfrage ein, um relevante Inhalte zu finden
2. Stellen Sie eine Frage, die basierend auf diesen Inhalten beantwortet werden soll
3. Passen Sie die Suchparameter an (Zeitraum, Chunk-Größe, etc.)
4. Verwenden Sie Schlagwörter für präzisere Ergebnisse
5. Aktivieren Sie die semantische Erweiterung, um ähnliche Begriffe einzubeziehen
6. Nutzen Sie die Zeitfenster-Suche für eine ausgewogene zeitliche Verteilung der Ergebnisse

### Schlagwort-Analyse

Die Anwendung bietet Tools zur Analyse von Schlagwörtern:

- Finden ähnlicher Wörter basierend auf dem Korpus
- Erweiterung boolescher Ausdrücke mit semantisch ähnlichen Begriffen
- Analyse der Häufigkeit von Begriffen im Korpus

### Modellauswahl

Sie können zwischen verschiedenen Sprachmodellen wählen:

- **HU-LLM**: Lokales Modell (kein API-Schlüssel erforderlich, HU-Netzwerk erforderlich)
- **OpenAI GPT-4o**: Leistungsstärkstes OpenAI-Modell (erfordert API-Schlüssel)
- **OpenAI GPT-3.5 Turbo**: Schnelleres OpenAI-Modell (erfordert API-Schlüssel)

## Hinweise und Einschränkungen

- Die Anwendung setzt eine Verbindung zum HU-Berlin ChromaDB (für die Vektordatenbank) und Ollama Embedding Service (für Sentence Embeddings) voraus
- Die FastText-Worteinbettungen werden lokal verwendet und müssen separat heruntergeladen werden
- Für die Nutzung von OpenAI-Modellen ist ein API-Schlüssel erforderlich
- Die Datengrundlage umfasst nur Spiegel-Artikel von 1948 bis 1979
- Das Starten der API ist nur für programmatischen Zugriff notwendig, die UI funktioniert direkt mit den Core-Modulen

## Fehlerbehebung

### Verbindungsprobleme

- Überprüfen Sie die Verbindungseinstellungen in der `.env`-Datei
- Stellen Sie sicher, dass Sie im HU-Netzwerk sind oder VPN verwenden
- Testen Sie die Verbindung mit `python src/utils/component_test.py`

### Probleme mit Worteinbettungen

- Stellen Sie sicher, dass die FastText-Modelle korrekt heruntergeladen und im `models/`-Verzeichnis platziert wurden
- Überprüfen Sie den Pfad in der `.env`-Datei: `WORD_EMBEDDING_MODEL_PATH`

## Lizenz

Dieses Projekt ist für die akademische Nutzung an der HU Berlin bestimmt. Die Daten des Spiegel-Archivs unterliegen den Nutzungsbedingungen des Anbieters.

## Kontakt

Für Fragen und Support wenden Sie sich bitte an Noah Baumann.
