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
    
    ## Tipps für optimale Ergebnisse
    
    1. **Präzise Suchanfragen**: Je genauer Ihre Suchanfrage, desto relevanter die Ergebnisse
    2. **Konkrete Fragen**: Formulieren Sie spezifische Fragen zu den gesuchten Inhalten
    3. **Schlagwort-Filterung**: Nutzen Sie die Filterung, um irrelevante Ergebnisse auszuschließen
    4. **Semantische Erweiterung**: Aktivieren Sie diese Option, um auch Texte mit ähnlichen Begriffen zu finden
    5. **Zeitfenster-Suche**: Besonders nützlich, um zeitliche Entwicklungen zu analysieren
    
    ## Anwendungsbeispiele
    
    - **Historische Analysen**: Wie hat sich die Berichterstattung zu einem Thema über die Zeit verändert?
    - **Medienkritik**: Wie wurden bestimmte Ereignisse im Spiegel dargestellt?
    - **Diskursanalyse**: Welche Begriffe und Konzepte wurden im Zusammenhang mit einem Thema verwendet?
    - **Ereignisrecherche**: Wie wurde über ein spezifisches historisches Ereignis berichtet?
    
    ## Datengrundlage
    
    Die Datenbank enthält Der Spiegel-Artikel aus den Jahren 1948 bis 1979, die aus dem Spiegel-Archiv gescraped wurden.
    """)