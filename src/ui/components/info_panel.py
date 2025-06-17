# src/ui/components/info_panel.py
"""
Updated info panel component reflecting the integrated agent search approach.
"""
import gradio as gr

def create_info_panel():
    """
    Create the updated info panel UI component.
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
    
    ## Hauptfunktionen
    
    - **Semantische Suche**: Findet Inhalte basierend auf Bedeutung, nicht nur nach Schlüsselwörtern
    - **Zeitfenster-Analyse**: Ausgewogene Abdeckung verschiedener historischer Perioden
    - **Flexible Textgrößen**: Optimieren Sie die Suche mit verschiedenen Chunk-Größen (500, 2000, 3000 Zeichen)
    - **Mehrere LLM-Optionen**: HU-LLM (lokal), OpenAI GPT-4o, Google Gemini Pro
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
    
    ## Wann welche Suchmethode verwenden?
    
    ### Standard-Suche ist ideal für:
    - **Schnelle Informationssuche**: Direkter Zugriff auf relevante Inhalte
    - **Explorative Recherche**: Überblick über verfügbare Materialien
    - **Bekannte Themen**: Wenn Sie wissen, wonach Sie suchen
    - **Große Ergebnismengen**: Wenn Sie viele Quellen durchgehen möchten
    - **Einfache Fragestellungen**: Faktische oder deskriptive Anfragen
    
    ### Agenten-Suche ist besser für:
    - **Komplexe Analysen**: Vielschichtige historische Fragestellungen
    - **Qualitative Auswahl**: Wenn die Güte der Quellen entscheidend ist
    - **Vergleichende Studien**: Ausgewogene Darstellung verschiedener Perspektiven
    - **Spezifische Forschungsfragen**: Gezielte wissenschaftliche Untersuchungen
    - **Zeitübergreifende Analysen**: Entwicklungen über mehrere Jahrzehnte
    - **Diskursanalysen**: Untersuchung von Narrativen und Deutungsmustern
    
    ## Anwendungsbeispiele
    
    ### Standard-Suche Beispiele:
    - **Ereignisrecherche**: "Berichterstattung über den Mauerbau 1961"
    - **Themenüberblick**: "Artikel über die Studentenbewegung"
    - **Aktuelle Suche**: "Wirtschaftspolitik in den 1970er Jahren"
    - **Schlagwortsuche**: "Berlin AND Mauer NOT Sowjetunion"
    
    ### Agenten-Suche Beispiele:
    - **Diskursanalyse**: "Wie wandelte sich die Darstellung der DDR zwischen den 1950er und 1970er Jahren?"
    - **Medienanalyse**: "Welche journalistischen Strategien verwendete der Spiegel bei politischen Skandalen?"
    - **Vergleichsstudien**: "Unterschiede in der Berichterstattung über SPD- und CDU-Politiker"
    - **Entwicklungsanalysen**: "Wie veränderte sich die Bewertung amerikanischer Politik im Kalten Krieg?"
    
    ## Bewertungs-Prompt Vorlagen (Agenten-Suche)
    
    Das System bietet spezialisierte Prompt-Vorlagen für verschiedene Analysezwecke:
    
    - **Agent Default**: Allgemeine historische Quellenbewertung
    - **Agent Media Analysis**: Medienwissenschaftliche Analyse von Sprache und Stil
    - **Agent Discourse Analysis**: Diskursanalytische Untersuchungen
    - **Agent Historical Context**: Historisch-kontextuelle Bewertungen
    - **Agent Political Analysis**: Politikwissenschaftliche Analysen
    - **Agent Social Cultural**: Sozial- und kulturgeschichtliche Fragestellungen
    
    Jede Vorlage kann individuell angepasst werden, um spezifische Forschungsinteressen zu berücksichtigen.
    
    ## Download-Funktionen
    
    ### Standard-Downloads (beide Suchmethoden):
    - **JSON**: Strukturierte Daten mit vollständigen Metadaten
    - **CSV**: Tabellenformat für weitere Analyse in Excel/SPSS
    
    ### Umfassender Agent-Download:
    - **Alle abgerufenen Texte**: Nicht nur die ausgewählten, sondern alle initial gefundenen
    - **KI-Bewertungen**: Scores und Begründungen für jeden Text
    - **Suchkonfiguration**: Vollständige Dokumentation der Suchparameter
    - **Zeitfenster-Zuordnungen**: Welcher Text aus welchem Zeitfenster stammt
    - **Evaluations-Metadaten**: Detaillierte Informationen zum Bewertungsprozess
    
    ## Technische Details
    
    ### Datengrundlage
    - **Zeitraum**: Der Spiegel-Artikel 1948-1979
    - **Textverarbeitung**: Artikel in semantisch durchsuchbare Abschnitte unterteilt
    - **Chunk-Größen**: 500, 2000 oder 3000 Zeichen pro Abschnitt
    
    ### KI-Modelle
    - **HU-LLM 1 & 3**: Lokale Modelle (HU-Netzwerk erforderlich)
    - **OpenAI GPT-4o**: Leistungsstärkste Option für komplexe Analysen
    - **Google Gemini Pro**: Großes Kontextfenster für umfangreiche Texte
    
    ### Einbettungsmodelle
    - **Ollama nomic-embed-text**: Für semantische Vektorsuche
    - **FastText**: Für Wortähnlichkeiten und semantische Erweiterung
    
    ## Tipps für optimale Ergebnisse
    
    ### Allgemeine Empfehlungen:
    1. **Präzise Beschreibungen**: Je spezifischer Ihre Inhaltsbeschreibung, desto relevanter die Ergebnisse
    2. **Angemessene Zeiträume**: Nicht zu große Spannen wählen (max. 10-15 Jahre)
    3. **Passende Chunk-Größe**: 3000 für Kontext, 2000 für Balance, 500 für Präzision
    
    ### Standard-Suche Optimierung:
    1. **Schlagwort-Kombinationen**: Nutzen Sie AND/OR für präzise Filterung
    2. **Semantische Erweiterung**: Aktivieren für bessere Trefferquote
    3. **Zeitfenster**: Bei langen Zeiträumen für ausgewogene Abdeckung
    
    ### Agenten-Suche Optimierung:
    1. **Passende Prompt-Vorlage**: Wählen Sie die Vorlage, die Ihrem Analyseziel entspricht
    2. **Ausgewogene Chunk-Zahlen**: Nicht zu wenig initial (min. 30), nicht zu viele final (max. 25)
    3. **Zeitfenster-Größe**: 3-7 Jahre für ausgewogene Abdeckung
    4. **Model-Wahl**: GPT-4o für komplexeste Analysen, HU-LLM für Standard-Bewertungen
    
    ## Häufige Anwendungsfälle
    
    ### Für Historiker:
    - **Quellenkritik**: Agent-Suche mit "Historical Context" Prompt
    - **Periodisierung**: Zeitfenster-Suche über mehrere Jahrzehnte
    - **Ereignisgeschichte**: Standard-Suche mit präzisen Zeiträumen
    
    ### Für Medienwissenschaftler:
    - **Sprachanalyse**: Agent-Suche mit "Media Analysis" Prompt
    - **Stil-Entwicklung**: Zeitfenster-Agent-Suche über längere Perioden
    - **Framing-Studien**: Vergleichende Agent-Suche verschiedener Themen
    
    ### Für Politikwissenschaftler:
    - **Meinungsbildung**: Agent-Suche mit "Political Analysis" Prompt
    - **Parteiendarstellung**: Schlagwort-gefilterte Vergleiche
    - **Politische Kultur**: Diskursanalytische Agent-Suche
    
    ### Für Studierende:
    - **Seminararbeiten**: Standard-Suche für Überblick, dann Agent-Suche für Vertiefung
    - **Quellensammlung**: Downloads für systematische Analyse
    - **Methodenlernen**: Vergleich verschiedener Suchstrategien
    
    ## Systemvoraussetzungen
    
    - **Netzwerk**: HU-Eduroam oder VPN für lokale LLM-Modelle
    - **Browser**: Moderne Browser (Chrome, Firefox, Safari, Edge)
    - **API-Schlüssel**: Nur für OpenAI/Gemini Modelle erforderlich (optional)
    
    ## Datenschutz und Nutzung
    
    - **Lokale Verarbeitung**: HU-LLM Modelle laufen lokal, keine externen Übertragungen
    - **Externe APIs**: OpenAI/Gemini nur bei expliziter Auswahl
    - **Archivdaten**: Unterliegen den Nutzungsbedingungen des Spiegel-Archivs
    - **Forschungszwecke**: System optimiert für akademische Nutzung
    """)