# src/ui/components/info_panel.py
"""
Info panel component for the Spiegel RAG application.
This component defines the UI elements for the information tab.
"""
import gradio as gr

def create_info_panel():
    """
    Create the info panel UI component.
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
    
    ## Hauptfunktionen
    
    - **Semantische Suche**: Findet Inhalte basierend auf ihrer Bedeutung, nicht nur nach Schlüsselwörtern
    - **Schlagwort-Filterung**: Verwenden Sie boolesche Operatoren (AND, OR, NOT) für präzise Filterung
    - **Semantische Erweiterung**: Findet und berücksichtigt automatisch ähnliche Begriffe
    - **Zeitfenster-Suche**: Analysiert Inhalte über verschiedene Zeitperioden hinweg
    - **Anpassbare Textgrößen**: Optimieren Sie die Suche mit verschiedenen Chunk-Größen
    - **Flexible LLM-Auswahl**: Wählen Sie zwischen HU-LLM (lokal) oder OpenAI-Modellen
    - **Zweistufiger Prozess**: Quellen gezielt abrufen und dann Fragen dazu stellen
    - **Agenten-basierte Suche**: Fortschrittliche mehrstufige Filterung mit LLM-gestützter Bewertung
    
    ## So nutzen Sie die Anwendung
    
    ### Standard-Suche (Zweistufiger Prozess)
    
    1. Geben Sie eine **Inhaltsbeschreibung** ein, was Sie im Archiv finden möchten
    2. Stellen Sie den **Zeitraum** ein (1948-1979)
    3. Nutzen Sie **Schlagwort-Filterung** mit booleschen Ausdrücken für präzisere Ergebnisse
    4. Aktivieren Sie die **Semantische Erweiterung** für ähnliche Begriffe
    5. Probieren Sie die **Zeitfenster-Suche** für eine ausgewogene zeitliche Abdeckung
    6. Klicken Sie auf **"Quellen abrufen"** - diese werden unter "Gefundene Texte" angezeigt
    
    Anschließend:
    
    1. Stellen Sie eine **konkrete Frage** zu den gefundenen Texten
    2. Wählen Sie ein **Sprachmodell** (HU-LLM oder OpenAI)
    3. Für OpenAI-Modelle ist ein **API-Schlüssel** erforderlich
    4. Passen Sie bei Bedarf den **System-Prompt** und die **Temperatur** an
    5. Klicken Sie auf **"Frage beantworten"**
    
    ### Agenten-basierte Suche (Fortgeschritten)
    
    Die Agenten-basierte Suche kombiniert Retrieval und Analyse in einem mehrstufigen Prozess:
    
    1. Geben Sie Ihre **Frage** ein
    2. Optional: Geben Sie eine **Inhaltsbeschreibung** ein, die sich von der Frage unterscheidet
       - Die Inhaltsbeschreibung definiert die Art der Inhalte, die gesucht werden sollen
       - Die Frage bestimmt, wie diese Inhalte bewertet und analysiert werden
    3. Konfigurieren Sie die **Filtereinstellungen**:
       - **Initiale Textmenge**: Wie viele Texte zunächst abgerufen werden (z.B. 100)
       - **Filterstufen**: Wie die Menge der Texte schrittweise reduziert wird (z.B. 50 → 20 → 10)
    4. Wählen Sie ein **LLM-Modell** für die Bewertung und Analyse
    5. Klicken Sie auf **"Agenten-Suche starten"**
    
    In diesem Modus:
    - Werden zunächst mehr Texte abgerufen als bei der Standard-Suche
    - Das LLM bewertet jeden Text hinsichtlich seiner Relevanz für Ihre Frage
    - Sie erhalten detaillierte Bewertungen und Begründungen für jeden ausgewählten Text
    - Die finale Antwort basiert auf den bestbewerteten Texten
    
    ## Tipps für optimale Ergebnisse
    
    1. **Präzise Suchanfragen**: Je genauer Ihre Inhaltsbeschreibung, desto relevanter die Quellen
    2. **Konkrete Fragen**: Formulieren Sie spezifische Fragen zu den gefundenen Texten
    3. **Schlagwort-Filterung**: Nutzen Sie boolesche Ausdrücke wie `berlin AND mauer NOT sowjet`
    4. **Semantische Erweiterung**: Findet automatisch verwandte Begriffe wie „DDR" für „Ostdeutschland"
    5. **Zeitfenster-Suche**: Besonders nützlich bei längeren Zeiträumen (>10 Jahre)
    6. **Agenten-Modus für komplexe Fragen**: Nutzen Sie den Agenten-Modus für analytische Fragen, die eine sorgfältigere Auswahl der Quellen erfordern
    7. **Unterschiedliche Inhaltsbeschreibung und Frage**: Im Agenten-Modus können Sie mit einer breiteren Inhaltsbeschreibung und einer spezifischen Frage noch bessere Ergebnisse erzielen
    
    ## Anwendungsbeispiele
    
    - **Historische Analysen**: Wie hat sich die Berichterstattung zur deutschen Teilung über die Jahre verändert?
    - **Medienkritik**: Wie wurden politische Ereignisse im Spiegel dargestellt?
    - **Diskursanalyse**: Welche Sprache und Konzepte wurden verwendet, um über den Kalten Krieg zu berichten?
    - **Ereignisrecherche**: Wie wurde über den Bau der Berliner Mauer berichtet?
    - **Vergleichende Analyse** (Agenten-Modus): Wie unterschied sich die Berichterstattung über westliche und östliche Politiker?
    - **Musteridentifikation** (Agenten-Modus): Welche Arten von Skandalen wurden im Spiegel behandelt und in welchem Tonfall?
    
    ## Wann welchen Modus verwenden?
    
    **Standard-Suche** ist ideal für:
    - Schnelle, direkte Informationssuche
    - Exploratives Durchsuchen des Archivs
    - Einfachere, faktische Fragen
    
    **Agenten-basierte Suche** ist besser für:
    - Komplexe, analytische Fragestellungen
    - Vergleichende Betrachtungen
    - Untersuchungen von Mustern oder Entwicklungen
    - Fragen, die tiefere Textverständnis erfordern
    - Situationen, in denen die Qualität der Quellenauswahl besonders wichtig ist
    
    ## Datengrundlage
    
    Die Datenbank enthält Der Spiegel-Artikel aus den Jahren 1948 bis 1979, die aus dem Spiegel-Archiv erschlossen wurden.
    Die Texte sind in Abschnitte (Chunks) unterteilt und semantisch durchsuchbar.
    """)