# src/ui/components/info_panel.py - Updated with new terminology and structure
"""
Updated info panel component with new terminology:
- Heuristik (instead of Quellen abrufen)
- Analyse (instead of Quellen analysieren)  
- LLM-Unterstützte Auswahl (instead of Agenten-Suche)
- Zeitintervall-Suche (instead of Zeitfenster-Suche)
"""
import gradio as gr

def create_info_panel():
    """
    Create the updated info panel UI component with new terminology and structure.
    """
    gr.Markdown("""
    # Über das SPIEGEL RAG-System
    
    Dieses System ermöglicht die Durchsuchung und Analyse von Der Spiegel-Artikeln aus den Jahren 1948 bis 1979 
    mithilfe von Retrieval-Augmented Generation (RAG).
    
    ## Was ist RAG?
    
    Retrieval-Augmented Generation ist ein Ansatz, der die Stärken von Suchsystemen mit 
    denen von großen Sprachmodellen kombiniert:
    
    1. **Heuristik**: Das System sucht zunächst relevante Textabschnitte aus dem Archiv
    2. **Analyse**: Ein Sprachmodell analysiert diese Abschnitte und generiert eine Antwort
    
    ## Zwei-Phasen-Ansatz
    
    Das System bietet einen strukturierten Workflow mit zwei parallelen Suchmethoden:
    
    ### Phase 1: Heuristik (Quellenauswahl)
    
    #### Standard-Suche (Schnell & Direkt)
    - **Vektorsimilarität**: Findet Inhalte basierend auf semantischer Ähnlichkeit
    - **Schlagwort-Filterung**: Boolesche Operatoren für präzise Filterung (immer aktiv)
    - **Semantische Erweiterung**: Automatische Berücksichtigung ähnlicher Begriffe mit Korpus-Häufigkeiten
    - **Zeitintervall-Suche**: Gleichmäßige zeitliche Verteilung für diakrone Narrative
    - **Geschwindigkeit**: Schnelle Ergebnisse für direkte Informationssuche
    
    #### LLM-Unterstützte Auswahl (KI-gestützte Bewertung)
    - **Intelligente Vorauswahl**: Zunächst mehr Quellen abrufen (z.B. 50 pro Zeitintervall)
    - **KI-Bewertung**: Sprachmodell bewertet jeden Text hinsichtlich der Relevanz
    - **Selektive Filterung**: Nur die besten Texte werden ausgewählt (z.B. 20 pro Zeitintervall)
    - **Transparente Begründungen**: Nachvollziehbare Erklärungen für jede Auswahl
    - **Zeitintervall-Integration**: Gleichmäßige Verteilung über verschiedene Perioden
    - **Anpassbare Bewertungskriterien**: Verschiedene Prompt-Vorlagen für unterschiedliche Analysezwecke
    - **Temperatur-Kontrolle**: Determinismus der KI-Bewertung konfigurierbar
    
    ### Phase 2: Analyse (Quellenverarbeitung)
    
    #### 1. Quellenauswahl
    - **Interaktive Auswahl**: Checkbox-basierte Auswahl aus gefundenen Quellen
    - **Vorauswahl**: Alle Quellen standardmäßig ausgewählt
    - **Übertragung**: Explizite Übertragung ausgewählter Quellen zur Analyse
    
    #### 2. User-Prompt formulieren
    - **Forschungsfrage**: Konkrete Fragestellung an die ausgewählten Quellen
    - **Flexibilität**: Unabhängig von der ursprünglichen Retrieval-Query
    
    #### 3. LLM-Auswählen
    - **Modellauswahl**: Verschiedene Sprachmodelle für unterschiedliche Anforderungen
    - **Temperatur-Kontrolle**: Determinismus der Antwortgenerierung einstellbar
    
    #### 4. System-Prompt
    - **Methodische Anleitung**: Steuerung der Analysemethodik
    - **Wissenschaftliche Standards**: Fokus auf Quellentreue und akademische Präzision
    - **Anpassbare Vorlagen**: Grundlegende Prompts für verschiedene Analyseansätze
    
    ## Verfügbare KI-Modelle
    
    Das System unterstützt verschiedene Sprachmodelle für unterschiedliche Anwendungszwecke:
    
    ### Lokale Modelle (HU-Netzwerk erforderlich)
    - **HU-LLM 1 & 3**: Schnelle, zuverlässige Modelle für Standard-Analysen
    - **DeepSeek R1 32B**: Fortschrittliches Reasoning-Modell für komplexe analytische Aufgaben
    
    ### Externe Modelle (API-Schlüssel erforderlich)
    - **OpenAI GPT-4o**: Vielseitiges, leistungsstarkes Modell
    - **Google Gemini 2.5 Pro**: Großes Kontextfenster für umfangreiche Textanalysen

    ## Hauptfunktionen
    
    - **Retrieval-Query-Optimierung**: Semantische Suche mit wenigen Phrasen und vielen Begriffen
    - **Zeitintervall-Analyse**: Gleichmäßige Abdeckung verschiedener historischer Perioden
    - **Flexible Chunking-Größen**: Optimieren Sie die Suche mit verschiedenen Chunk-Größen (500, 2000, 3000 Zeichen)
    - **Vier leistungsstarke LLM-Optionen**: HU-LLM, DeepSeek R1, OpenAI GPT-4o, Google Gemini 2.5 Pro
    - **Umfassende Downloads**: JSON/CSV-Export mit Metadaten und KI-Bewertungen, TXT-Export für Analysen
    - **Integrierter Workflow**: Strukturierte Heuristik → Quellenauswahl → Analyse
    
    ## So nutzen Sie die Anwendung
    
    ### 1. Heuristik durchführen
    
    **Schritt 1: Suchmethode wählen**
    - **Standard-Suche**: Für schnelle, direkte Informationssuche
    - **LLM-Unterstützte Auswahl**: Für sorgfältige, KI-gestützte Quellenauswahl
    
    **Schritt 2: Suchparameter konfigurieren**
    - Geben Sie eine **Retrieval-Query** ein (wenige Phrasen, viele Begriffe)
    - Stellen Sie den **Zeitraum** ein (1948-1979)
    - Wählen Sie die **Chunking-Größe** (500/2000/3000 Zeichen)
    
    **Für Standard-Suche zusätzlich:**
    - **Anzahl Ergebnisse** festlegen (1-50 gesamt oder pro Zeitintervall)
    - Optional: **Schlagwort-Filterung** mit booleschen Ausdrücken (immer strikte Filterung)
    - Optional: **Semantische Erweiterung** für ähnliche Begriffe mit Häufigkeitsanalyse
    - Optional: **Zeitintervall-Suche** für ausgewogene zeitliche Abdeckung
    
    **Für LLM-Unterstützte Auswahl zusätzlich:**
    - **Zeit-Intervalle** konfigurieren (standardmäßig aktiviert)
    - **Chunks pro Intervall** einstellen (Initial → Final, z.B. 50 → 20)
    - **KI-Modell** für Bewertung wählen mit **Temperatur-Kontrolle**
    - **Bewertungs-Prompt** anpassen (Standard, Negative Reranking, Kontextuelle Bewertung)
    
    **Schritt 3: Heuristik starten**
    - Bei Standard-Suche: Sofortige Ergebnisse
    - Bei LLM-Unterstützter Auswahl: Fortschrittsanzeige mit Zeitintervall-Updates
    
    ### 2. Quellenauswahl treffen
    
    - **Gefundene Texte** werden mit Checkboxen angezeigt
    - **Interaktive Auswahl**: Standardmäßig alle ausgewählt, individuell an-/abwählbar
    - **Übertragung**: "Auswahl in Analyse übertragen" Button aktiviert Analyse-Phase
    
    ### 3. Analyse durchführen
    
    - **User-Prompt formulieren**: Konkrete Forschungsfrage stellen
    - **LLM-Modell** auswählen mit **Temperatur-Einstellung**
    - **System-Prompt** konfigurieren für methodische Steuerung
    - **Analyse starten** und Ergebnisse als TXT herunterladen
    
    ## Technische Details
    
    ### Datengrundlage
    - **Zeitraum**: Der Spiegel-Artikel 1948-1979
    - **Textverarbeitung**: Artikel in semantisch durchsuchbare Abschnitte unterteilt
    - **Chunking-Größen**: 500, 2000 oder 3000 Zeichen pro Abschnitt
    
    ### KI-Modelle im Detail
    - **HU-LLM 1 & 3**: Lokale Modelle für Standard-Aufgaben (HU-Netzwerk erforderlich)
    - **DeepSeek R1 32B**: Reasoning-Modell via Ollama für analytische Aufgaben (HU-Netzwerk erforderlich)
    - **OpenAI GPT-4o**: Vielseitigstes externes Modell für alle Aufgabentypen
    - **Google Gemini 2.5 Pro**: Großes Kontextfenster für umfangreiche Texte
    
    ### Einbettungsmodelle
    - **Ollama nomic-embed-text**: Für semantische Vektorsuche
    - **FastText**: Für Wortähnlichkeiten und semantische Erweiterung mit Korpus-Häufigkeiten
    
    ## Tipps für optimale Ergebnisse
    
    ### Retrieval-Query-Optimierung:
    1. **Wenige Phrasen, viele Begriffe**: "Berliner Mauer Grenze DDR" statt "Wie wurde die Berliner Mauer dargestellt?"
    2. **Zeitgenössische Sprache**: Nutzen Sie Begriffe aus der Zeit für bessere Retrieval-Qualität
    3. **Semantische Erweiterung**: Aktivieren Sie diese für breitere Begriffsabdeckung
    4. **Zeitintervall-Suche**: Für diakrone Narrative und ausgewogene zeitliche Abdeckung
    
    ### Analyse-Optimierung:
    1. **Chunking-Größe**: 3000 für Kontext, 2000 für Balance, 500 für Präzision
    2. **Modell-Matching**: Einfache Aufgaben → HU-LLM, komplexe Analysen → DeepSeek R1
    3. **Temperatur**: Niedrigere Werte (0.1-0.3) für konsistentere, fokussiertere Antworten
    4. **System-Prompt**: Fokus auf wissenschaftliche Methodik und Quellentreue
    
    ### LLM-Unterstützte Auswahl:
    1. **Negative Reranking**: Für kritische Bewertung mit Pro-/Contra-Argumentation
    2. **Kontextuelle Bewertung**: Für historische Einordnung und Quellenwert-Bewertung
    3. **Temperatur-Kontrolle**: Niedrigere Werte für konsistentere Bewertungen
    
    ## Systemvoraussetzungen
    
    - **Netzwerk**: HU-Eduroam oder VPN für lokale LLM-Modelle (HU-LLM 1, 3, DeepSeek R1)
    - **Browser**: Moderne Browser (Chrome, Firefox, Safari, Edge)
    - **API-Schlüssel**: Nur für OpenAI/Gemini Modelle erforderlich (optional)
    
    ## Datenschutz und Nutzung
    
    - **Lokale Verarbeitung**: HU-LLM und DeepSeek R1 laufen lokal, keine externen Übertragungen
    - **Externe APIs**: OpenAI/Gemini nur bei expliziter Auswahl
    - **Archivdaten**: Unterliegen den Nutzungsbedingungen des Spiegel-Archivs
    - **Forschungszwecke**: System optimiert für akademische Nutzung
    
    ## Begriffserklärungen
    
    - **Heuristik**: Systematische Quellensuche und -bewertung
    - **Retrieval-Query**: Suchanfrage zur Quellenidentifikation
    - **Chunking-Größe**: Länge der Textabschnitte für die Verarbeitung
    - **Zeitintervall-Suche**: Zeitlich ausgewogene Quellenauswahl
    - **LLM-Unterstützte Auswahl**: KI-gestützte Quellenbewertung
    - **User-Prompt**: Konkrete Forschungsfrage an die Quellen
    - **System-Prompt**: Methodische Anweisungen an das KI-System
    - **Temperatur**: Determinismus-Grad der KI-Generierung (0 = deterministisch, 1 = kreativ)
                  """)