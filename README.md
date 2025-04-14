
# Der Spiegel RAG System (1948-1979)

Ein Retrieval Augmented Generation (RAG) System zur Analyse und Durchsuchung des Spiegel-Archivs der Jahrgänge 1948-1979.

## Übersicht

Dieses System ermöglicht die semantische Suche, Analyse und KI-gestützte Auswertung von Artikeln aus dem Spiegel-Archiv. Es kombiniert Vektordatenbanktechnologie mit großen Sprachmodellen, um relevante Textpassagen zu finden und basierend auf diesen Informationen qualitativ hochwertige Antworten zu generieren.

### Hauptfunktionen

* **Semantische Suche** : Finde relevante Inhalte basierend auf Bedeutung, nicht nur auf Schlagwörtern
* **Zeitfenster-Suche** : Analysiere Inhalte über verschiedene Zeitperioden hinweg
* **Schlagwort-Filterung** : Verwende boolesche Ausdrücke (AND, OR, NOT) für präzise Filterung
* **Semantische Erweiterung** : Automatische Erweiterung von Suchbegriffen mit ähnlichen Wörtern
* **KI-gestützte Analyse** : Generiere Zusammenfassungen und Analysen basierend auf gefundenen Texten
* **Mehrsprachige Modelle** : Unterstützung für lokale (HU-LLM) und externe (OpenAI) Sprachmodelle
* **Zweistufiges Retrieval & Analyse** : Zuerst relevante Quellen abrufen, dann gezielt Fragen dazu stellen
* **Agenten-basierte Suche** : Fortschrittlicher Hybrid-Ansatz mit mehrstufiger LLM-gesteuerter Filterung

## Systemanforderungen

* Python 3.8 oder höher
* Internetzugang über HU-Eduroam oder VPN für die Verbindung zum ChromaDB und Ollama Embedding Service
* Optional: OpenAI API-Schlüssel für GPT-4o oder GPT-3.5 Turbo

## Installation

### 1. Repository klonen

```bash
git clone [repository-url]
cd spiegel_rag
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

```bash
# Automatische Installation mit dem bereitgestellten Skript
python models/model_data_import.py
```

Alternativ:

1. Laden Sie die Einbettungen von der angegebenen URL herunter: [https://box.hu-berlin.de/f/34c17bfd74b84454b276/?dl=1](https://box.hu-berlin.de/f/34c17bfd74b84454b276/?dl=1)
2. Entpacken Sie die Dateien in den `models/` Ordner

### 5. Daten herunterladen (optional)

Die Anwendung verbindet sich standardmäßig mit einem Remote-Service für den Zugriff auf die Vektordatenbank. Wenn Sie die Daten lokal analysieren möchten, können Sie die CSV-Dateien herunterladen:

```bash
# Automatische Installation mit dem bereitgestellten Skript
python data/spiegel_data_import.py
```

Alternativ:

1. Laden Sie die CSV-Dateien von [https://box.hu-berlin.de/f/1664f2bc4fa9439a8590/?dl=1](https://box.hu-berlin.de/f/1664f2bc4fa9439a8590/?dl=1) herunter
2. Entpacken Sie sie in den `data/csv_daten/` Ordner

**Hinweis zu verfügbaren ChromaDB-Kollektionen:**
Das System ist so konfiguriert, dass es die folgenden zwei existierenden ChromaDB-Kollektionen verwendet:

1. `recursive_chunks_3000_300_TH_cosine_nomic-embed-text`
2. `recursive_chunks_2000_400_TH_cosine_nomic-embed-text`

## Ausführen der Anwendung

### Starten der Benutzeroberfläche

```bash
python src/ui/app.py
```

Die Anwendung ist dann unter [http://localhost:7860](http://localhost:7860/) verfügbar.

## Komponententests

Testen Sie die Kernkomponenten mit dem Diagnose-Tool:

```bash
python src/utils/component_test.py
```

## Nutzungsanleitung

Die Anwendung bietet zwei Hauptansätze zur Informationsgewinnung:

### 1. Standard-RAG (Zweistufiger Prozess)

#### Schritt 1: Quellen abrufen

1. Geben Sie eine Inhaltsbeschreibung ein - was möchten Sie im Archiv finden?
2. Legen Sie den Zeitraum fest (zwischen 1948 und 1979)
3. Konfigurieren Sie optional die Schlagwortfilterung mit booleschen Ausdrücken
4. Aktivieren Sie bei Bedarf die semantische Erweiterung, um ähnliche Begriffe einzubeziehen
5. Nutzen Sie die Zeitfenster-Suche für eine ausgewogene zeitliche Verteilung der Ergebnisse
6. Klicken Sie auf "Quellen abrufen"

#### Schritt 2: Quellen analysieren

1. Stellen Sie eine Frage über die gefundenen Texte
2. Wählen Sie das zu verwendende Sprachmodell:
   * HU-LLM (lokales Modell, kein API-Schlüssel benötigt)
   * OpenAI GPT-4o oder GPT-3.5 Turbo (API-Schlüssel erforderlich)
3. Passen Sie bei Bedarf die LLM-Parameter wie System-Prompt, Temperatur und maximale Antwortlänge an
4. Klicken Sie auf "Frage beantworten"

### 2. Agenten-basierte Suche (Fortgeschritten)

Die Agenten-basierte Suche kombiniert Retrieval und Analyse in einem mehrstufigen Prozess:

1. Geben Sie Ihre Frage ein
2. Optional: Geben Sie eine Inhaltsbeschreibung ein (falls unterschiedlich von der Frage)
3. Konfigurieren Sie die Filtereinstellungen:
   * Initiale Textmenge (z.B. 100 Chunks)
   * Filterstufen (z.B. 50 → 20 → 10)
4. Wählen Sie das zu verwendende Sprachmodell
5. Klicken Sie auf "Agenten-Suche starten"

Im Agenten-Modus:

- Werden zunächst mehr Texte abgerufen
- Das LLM bewertet jeden Text hinsichtlich seiner Relevanz für Ihre Frage
- Sie erhalten detaillierte Bewertungen und Begründungen für die ausgewählten Texte
- Die finale Antwort basiert auf den bestbewerteten Texten

#### Wann den Agenten-Modus verwenden?

Der Agenten-Modus ist besonders hilfreich für:

- Komplexe, analytische Fragestellungen
- Vergleichende Betrachtungen
- Untersuchungen von Mustern oder Entwicklungen
- Fragen, die tieferes Textverständnis erfordern

### Schlagwort-Analyse

Im Tab "Schlagwort-Analyse" können Sie:

* Ähnliche Wörter zu einem Suchbegriff finden
* Die Ergebnisse für komplexere Suchanfragen nutzen

## Architektur

Die Anwendung besteht aus folgenden Hauptkomponenten:

### Kern-Module

* **RAG Engine** (`src/core/rag_engine.py`): Hauptmodul, das die Suche und Antworterstellung koordiniert
* **Retrieval Agent** (`src/core/retrieval_agent.py`): Implementiert die agenten-basierte mehrstufige Filterung
* **Vector Store** (`src/core/vector_store.py`): Interface zur externen ChromaDB-Vektordatenbank
* **LLM Service** (`src/core/llm_service.py`): Interface zu verschiedenen Sprachmodellen (HU-LLM oder OpenAI)
* **Embedding Service** (`src/core/embedding_service.py`): Lokaler Worteinbettungs-Service für semantische Ähnlichkeitssuche mit FastText

Der Datenfluss in der Anwendung:

1. Der Benutzer gibt eine Suchanfrage in der UI ein
2. Je nach gewähltem Modus:
   - **Standard-RAG**: Die RAG Engine sendet eine Anfrage an den externen ChromaDB-Service, um relevante Textabschnitte zu finden
   - **Agenten-Modus**: Der Retrieval Agent ruft zunächst mehr Texte ab und lässt diese vom LLM bewerten und filtern
3. Der externe Ollama-Service liefert die Sentence Embeddings für die Ähnlichkeitssuche
4. Der lokale Embedding Service erweitert Schlagwörter mit semantisch ähnlichen Begriffen
5. Die gefundenen Texte werden angezeigt und zur Analyse verwendet
6. Das ausgewählte Sprachmodell (HU-LLM oder OpenAI) generiert eine fundierte Antwort

### UI-Komponenten

* **Such-Panel** (`src/ui/components/search_panel.py`): Interface zum Abrufen von Quellen
* **Frage-Panel** (`src/ui/components/question_panel.py`): Interface zum Stellen von Fragen zu gefundenen Quellen
* **Ergebnis-Panel** (`src/ui/components/results_panel.py`): Anzeige der Analyseergebnisse
* **Agenten-Panel** (`src/ui/components/agent_panel.py`): Interface für die agenten-basierte Suche
* **Agenten-Ergebnis-Panel** (`src/ui/components/agent_results_panel.py`): Anzeige der Agenten-Analyseergebnisse
* **Schlagwort-Analyse-Panel** (`src/ui/components/keyword_analysis_panel.py`): Tools zur Analyse von Schlagwörtern

## Hinweise und Einschränkungen

* Die Anwendung setzt eine Verbindung zum HU-Berlin ChromaDB (für die Vektordatenbank) und Ollama Embedding Service (für Sentence Embeddings) voraus
* Die FastText-Worteinbettungen werden lokal verwendet und müssen separat heruntergeladen werden
* Für die Nutzung von OpenAI-Modellen ist ein API-Schlüssel erforderlich
* Die Datengrundlage umfasst nur Spiegel-Artikel von 1948 bis 1979
* Der Agenten-Modus benötigt aufgrund der mehrstufigen Analyse mehr Zeit als die Standard-Suche

## Fehlerbehebung

### Verbindungsprobleme

* Überprüfen Sie die Verbindungseinstellungen in der `.env`-Datei
* Stellen Sie sicher, dass Sie im HU-Netzwerk sind oder VPN verwenden
* Testen Sie die Verbindung mit `python src/utils/component_test.py`

### Probleme mit Worteinbettungen

* Stellen Sie sicher, dass die FastText-Modelle korrekt heruntergeladen und im `models/`-Verzeichnis platziert wurden
* Überprüfen Sie den Pfad in der `.env`-Datei: `WORD_EMBEDDING_MODEL_PATH`

## Lizenz

Dieses Projekt ist für die akademische Nutzung an der HU Berlin bestimmt. Die Daten des Spiegel-Archivs unterliegen den Nutzungsbedingungen des Anbieters.

## Kontakt

Für Fragen und Support wenden Sie sich bitte an Noah Baumann.
