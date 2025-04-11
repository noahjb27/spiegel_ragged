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
    
    ## So nutzen Sie die Anwendung
    
    ### Schritt 1: Quellen abrufen
    
    1. Geben Sie eine **Inhaltsbeschreibung** ein, was Sie im Archiv finden möchten
    2. Stellen Sie den **Zeitraum** ein (1948-1979)
    3. Nutzen Sie **Schlagwort-Filterung** mit booleschen Ausdrücken für präzisere Ergebnisse
    4. Aktivieren Sie die **Semantische Erweiterung** für ähnliche Begriffe
    5. Probieren Sie die **Zeitfenster-Suche** für eine ausgewogene zeitliche Abdeckung
    6. Klicken Sie auf **"Quellen abrufen"** - diese werden unter "Gefundene Texte" angezeigt
    
    ### Schritt 2: Quellen analysieren
    
    1. Stellen Sie eine **konkrete Frage** zu den gefundenen Texten
    2. Wählen Sie ein **Sprachmodell** (HU-LLM oder OpenAI)
    3. Für OpenAI-Modelle ist ein **API-Schlüssel** erforderlich
    4. Passen Sie bei Bedarf den **System-Prompt** und die **Temperatur** an
    5. Klicken Sie auf **"Frage beantworten"**
    
    ## Tipps für optimale Ergebnisse
    
    1. **Präzise Suchanfragen**: Je genauer Ihre Inhaltsbeschreibung, desto relevanter die Quellen
    2. **Konkrete Fragen**: Formulieren Sie spezifische Fragen zu den gefundenen Texten
    3. **Schlagwort-Filterung**: Nutzen Sie boolesche Ausdrücke wie `berlin AND mauer NOT sowjet`
    4. **Semantische Erweiterung**: Findet automatisch verwandte Begriffe wie „DDR" für „Ostdeutschland"
    5. **Zeitfenster-Suche**: Besonders nützlich bei längeren Zeiträumen (>10 Jahre)
    
    ## Anwendungsbeispiele
    
    - **Historische Analysen**: Wie hat sich die Berichterstattung zur deutschen Teilung über die Jahre verändert?
    - **Medienkritik**: Wie wurden politische Ereignisse im Spiegel dargestellt?
    - **Diskursanalyse**: Welche Sprache und Konzepte wurden verwendet, um über den Kalten Krieg zu berichten?
    - **Ereignisrecherche**: Wie wurde über den Bau der Berliner Mauer berichtet?
    
    ## Datengrundlage
    
    Die Datenbank enthält Der Spiegel-Artikel aus den Jahren 1948 bis 1979, die aus dem Spiegel-Archiv erschlossen wurden.
    Die Texte sind in Abschnitte (Chunks) unterteilt und semantisch durchsuchbar.
    """)