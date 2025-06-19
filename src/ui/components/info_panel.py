# src/ui/components/info_panel.py - Updated with DeepSeek R1 information
"""
Updated info panel component reflecting the integrated agent search approach and DeepSeek R1 model.
"""
import gradio as gr

def create_info_panel():
    """
    Create the updated info panel UI component with DeepSeek R1 information.
    """
    gr.Markdown("""
    # Über das Spiegel RAG-System
    
    Dieses System ermöglicht die Durchsuchung und Analyse von Der Spiegel-Artikeln aus den Jahren 1948 bis 1979 
    mithilfe von Retrieval-Augmented Generation (RAG).
    
    ## Was ist RAG?
    
    Retrieval-Augmented Generation ist ein Ansatz, der die Stärken von Suchsystemen mit 
    denen von großen Sprachmodellen kombiniert:
    
    1. **Retrieval**: Das System sucht zunächst relevante Textabschnitte aus dem Archiv
    2. **Generation**: Ein Sprachmodell analysiert diese Abschnitte und generiert eine Antwort
    
    ## Zwei Suchmethoden
    
    Das System bietet zwei parallele Ansätze zur Quellenauswahl:
    
    ### Standard-Suche (Schnell & Direkt)
    - **Vektorsimilarität**: Findet Inhalte basierend auf semantischer Ähnlichkeit
    - **Schlagwort-Filterung**: Boolesche Operatoren (AND, OR, NOT) für präzise Filterung
    - **Semantische Erweiterung**: Automatische Berücksichtigung ähnlicher Begriffe
    - **Zeitfenster-Suche**: Ausgewogene Abdeckung verschiedener Zeitperioden
    - **Geschwindigkeit**: Schnelle Ergebnisse für direkte Informationssuche
    
    ### Agenten-Suche (KI-gestützte Bewertung)
    - **Intelligente Vorauswahl**: Zunächst mehr Quellen abrufen (z.B. 50 pro Zeitfenster)
    - **KI-Bewertung**: Sprachmodell bewertet jeden Text hinsichtlich der Relevanz
    - **Selektive Filterung**: Nur die besten Texte werden ausgewählt (z.B. 20 pro Zeitfenster)
    - **Transparente Begründungen**: Nachvollziehbare Erklärungen für jede Auswahl
    - **Zeitfenster-Integration**: Gleichmäßige Verteilung über verschiedene Perioden
    - **Anpassbare Bewertungskriterien**: Verschiedene Prompt-Vorlagen für unterschiedliche Analysezwecke
    
    ## Verfügbare KI-Modelle
    
    Das System unterstützt verschiedene Sprachmodelle für unterschiedliche Anwendungszwecke:
    
    ### Lokale Modelle (HU-Netzwerk erforderlich)
    - **HU-LLM 1 & 3**: Schnelle, zuverlässige Modelle für Standard-Analysen
    - **DeepSeek R1 32B**: Fortschrittliches Reasoning-Modell für komplexe analytische Aufgaben
    
    ### Externe Modelle (API-Schlüssel erforderlich)
    - **OpenAI GPT-4o**: Vielseitiges, leistungsstarkes Modell
    - **Google Gemini Pro**: Großes Kontextfenster für umfangreiche Textanalysen

    ## Hauptfunktionen
    
    - **Semantische Suche**: Findet Inhalte basierend auf Bedeutung, nicht nur nach Schlüsselwörtern
    - **Zeitfenster-Analyse**: Ausgewogene Abdeckung verschiedener historischer Perioden
    - **Flexible Textgrößen**: Optimieren Sie die Suche mit verschiedenen Chunk-Größen (500, 2000, 3000 Zeichen)
    - **Vier leistungsstarke LLM-Optionen**: HU-LLM, DeepSeek R1, OpenAI GPT-4o, Google Gemini Pro
    - **Umfassende Downloads**: JSON/CSV-Export mit Metadaten und KI-Bewertungen
    - **Integrierter Workflow**: Beide Suchmethoden führen nahtlos zur Analyse
    
    ## So nutzen Sie die Anwendung
    
    ### 1. Quellen abrufen
    
    **Schritt 1: Suchmethode wählen**
    - **Standard-Suche**: Für schnelle, direkte Informationssuche
    - **Agenten-Suche**: Für sorgfältige, KI-gestützte Quellenauswahl
    
    **Schritt 2: Suchparameter konfigurieren**
    - Geben Sie eine **Inhaltsbeschreibung** ein
    - Stellen Sie den **Zeitraum** ein (1948-1979)
    - Wählen Sie die **Textgröße** (Chunk-Größe)
    
    **Für Standard-Suche zusätzlich:**
    - **Anzahl Ergebnisse** festlegen (1-50)
    - Optional: **Schlagwort-Filterung** mit booleschen Ausdrücken
    - Optional: **Semantische Erweiterung** für ähnliche Begriffe
    - Optional: **Zeitfenster-Suche** für ausgewogene zeitliche Abdeckung
    
    **Für Agenten-Suche zusätzlich:**
    - **Zeitfenster** konfigurieren (standardmäßig aktiviert)
    - **Chunks pro Fenster** einstellen (Initial → Final, z.B. 50 → 20)
    - **KI-Modell** für Bewertung wählen 
    - **Bewertungs-Prompt** anpassen (vordefinierte Vorlagen verfügbar)
    
    **Schritt 3: Suche starten**
    - Bei Standard-Suche: Sofortige Ergebnisse
    - Bei Agenten-Suche: Fortschrittsanzeige mit Zeitfenster-Updates
    
    ### 2. Ergebnisse prüfen und herunterladen
    
    - **Gefundene Texte** werden automatisch angezeigt
    - **Download-Optionen**:
      - **JSON/CSV**: Standard-Downloads für alle Suchmethoden
      - **Umfassender Agent-Download**: Zusätzlich bei Agenten-Suche mit allen abgerufenen Texten und KI-Bewertungen
    
    ### 3. Quellen analysieren
    
    - Stellen Sie eine **konkrete Frage** zu den gefundenen Texten
    - Wählen Sie ein **Sprachmodell** für die Analyse 
    - Passen Sie bei Bedarf **System-Prompt** und **Parameter** an
    - Erhalten Sie eine **fundierte Antwort** basierend auf den ausgewählten Quellen
    
    ## Technische Details
    
    ### Datengrundlage
    - **Zeitraum**: Der Spiegel-Artikel 1948-1979
    - **Textverarbeitung**: Artikel in semantisch durchsuchbare Abschnitte unterteilt
    - **Chunk-Größen**: 500, 2000 oder 3000 Zeichen pro Abschnitt
    
    ### KI-Modelle im Detail
    - **HU-LLM 1 & 3**: Lokale Modelle für Standard-Aufgaben (HU-Netzwerk erforderlich)
    - **DeepSeek R1 32B**: Reasoning-Modell via Ollama für analytische Aufgaben (HU-Netzwerk erforderlich)
    - **OpenAI GPT-4o**: Vielseitigstes externes Modell für alle Aufgabentypen
    - **Google Gemini Pro**: Großes Kontextfenster für umfangreiche Texte
    
    ### Einbettungsmodelle
    - **Ollama nomic-embed-text**: Für semantische Vektorsuche
    - **FastText**: Für Wortähnlichkeiten und semantische Erweiterung
    
    ## Tipps für optimale Ergebnisse
    
    ### Allgemeine Empfehlungen:
    1. **Präzise Beschreibungen**: Je spezifischer Ihre Inhaltsbeschreibung, desto relevanter die Ergebnisse
    2. **Angemessene Zeiträume**: Nicht zu große Spannen wählen (max. 10-15 Jahre)
    3. **Passende Chunk-Größe**: 3000 für Kontext, 2000 für Balance, 500 für Präzision
    4. **Modell-Matching**: Einfache Aufgaben → HU-LLM, komplexe Analysen → DeepSeek R1
    
    ## Systemvoraussetzungen
    
    - **Netzwerk**: HU-Eduroam oder VPN für lokale LLM-Modelle (HU-LLM 1, 3, DeepSeek R1)
    - **Browser**: Moderne Browser (Chrome, Firefox, Safari, Edge)
    - **API-Schlüssel**: Nur für OpenAI/Gemini Modelle erforderlich (optional)
    
    ## Datenschutz und Nutzung
    
    - **Lokale Verarbeitung**: HU-LLM und DeepSeek R1 laufen lokal, keine externen Übertragungen
    - **Externe APIs**: OpenAI/Gemini nur bei expliziter Auswahl
    - **Archivdaten**: Unterliegen den Nutzungsbedingungen des Spiegel-Archivs
    - **Forschungszwecke**: System optimiert für akademische Nutzung
    
                  """)