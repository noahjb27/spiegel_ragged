# src/ui/app.py - Simplified unified UI
"""
Unified UI that combines all search modes into a single interface.
Reduces code duplication and improves user experience.
"""
import gradio as gr
from typing import Dict, Any, List, Tuple, Optional
import logging
import os
import sys 

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.engine import SpiegelRAG
from src.core.search.strategies import (
    StandardSearchStrategy, 
    TimeWindowSearchStrategy, 
    AgentSearchStrategy,
    SearchConfig
)


from src.services.search_service import SearchService
from src.config import settings

logger = logging.getLogger(__name__)


class UnifiedSearchUI:
    """Unified search interface with mode selection"""
    
    def __init__(self):
        self.search_service = SearchService()
        
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        
        with gr.Blocks(title="Der Spiegel RAG System") as app:
            gr.Markdown("""
            # Der Spiegel RAG System (1948-1979)
            
            Durchsuchen und analysieren Sie das historische Spiegel-Archiv mit KI-Unterstützung.
            """)
            
            # State management
            search_result_state = gr.State(None)
            
            with gr.Tab("Suche & Analyse"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Search mode selection
                        search_mode = gr.Radio(
                            choices=["Standard", "Zeitfenster", "Agent"],
                            value="Standard",
                            label="Suchmodus",
                            info="Wählen Sie die Suchmethode"
                        )
                        
                        # Common search parameters
                        content_description = gr.Textbox(
                            label="Inhaltsbeschreibung",
                            placeholder="Was möchten Sie im Archiv finden?",
                            lines=2
                        )
                        
                        with gr.Row():
                            year_start = gr.Slider(
                                minimum=settings.MIN_YEAR,
                                maximum=settings.MAX_YEAR,
                                value=1960,
                                step=1,
                                label="Von Jahr"
                            )
                            year_end = gr.Slider(
                                minimum=settings.MIN_YEAR,
                                maximum=settings.MAX_YEAR,
                                value=1970,
                                step=1,
                                label="Bis Jahr"
                            )
                        
                        chunk_size = gr.Dropdown(
                            choices=settings.AVAILABLE_CHUNK_SIZES,
                            value=settings.DEFAULT_CHUNK_SIZE,
                            label="Textgröße"
                        )
                        
                        # Optional keyword filtering
                        with gr.Accordion("Erweiterte Optionen", open=False):
                            keywords = gr.Textbox(
                                label="Schlagwörter",
                                placeholder="z.B. berlin AND mauer",
                                lines=1
                            )
                            
                            top_k = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label="Anzahl Ergebnisse"
                            )
                            
                            enforce_keywords = gr.Checkbox(
                                label="Strikte Filterung",
                                value=True
                            )
                        
                        # Mode-specific options
                        with gr.Group(visible=False) as timewindow_options:
                            window_size = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Fenstergröße (Jahre)"
                            )
                        
                        with gr.Group(visible=False) as agent_options:
                            initial_count = gr.Slider(
                                minimum=20,
                                maximum=200,
                                value=100,
                                step=10,
                                label="Initiale Textmenge"
                            )
                            
                            gr.Markdown("**Filterstufen** (absteigend)")
                            filter_stages = gr.CheckboxGroup(
                                choices=["50", "30", "20", "10", "5"],
                                value=["50", "20", "10"],
                                label="Behalte jeweils X Texte"
                            )
                        
                        search_btn = gr.Button("🔍 Suchen", variant="primary")
                        
                    with gr.Column(scale=2):
                        # Search status and results
                        search_status = gr.Markdown("Bereit zur Suche...")
                        
                        # Search results preview
                        with gr.Accordion("Gefundene Texte", open=False) as results_accordion:
                            search_results_preview = gr.Markdown()
                        
                        # Analysis section
                        with gr.Group(visible=False) as analysis_section:
                            gr.Markdown("### Analyse")
                            
                            question = gr.Textbox(
                                label="Ihre Frage",
                                placeholder="Was möchten Sie über die gefundenen Texte wissen?",
                                lines=2
                            )
                            
                            with gr.Row():
                                model_choice = gr.Radio(
                                    choices=["hu-llm", "gpt-4o", "gpt-3.5-turbo"],
                                    value="hu-llm",
                                    label="Modell"
                                )
                                
                                temperature = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    value=0.3,
                                    step=0.1,
                                    label="Temperatur"
                                )
                            
                            with gr.Row(visible=False) as api_key_row:
                                openai_api_key = gr.Textbox(
                                    label="OpenAI API Key",
                                    type="password"
                                )
                            
                            analyze_btn = gr.Button("💡 Analysieren", variant="primary")
                            
                            # Analysis results
                            analysis_output = gr.Markdown()
            
            with gr.Tab("Schlagwort-Analyse"):
                # Simplified keyword analysis
                with gr.Row():
                    keyword_input = gr.Textbox(
                        label="Wort eingeben",
                        placeholder="z.B. mauer"
                    )
                    find_similar_btn = gr.Button("Ähnliche Wörter finden")
                
                similar_words_output = gr.Markdown()
            
            with gr.Tab("Hilfe"):
                gr.Markdown(self._get_help_text())
            
            # Event handlers
            search_mode.change(
                self._update_mode_options,
                inputs=[search_mode],
                outputs=[timewindow_options, agent_options]
            )
            
            model_choice.change(
                lambda m: gr.update(visible=m.startswith("gpt")),
                inputs=[model_choice],
                outputs=[api_key_row]
            )
            
            search_btn.click(
                self._perform_search,
                inputs=[
                    search_mode, content_description, year_start, year_end,
                    chunk_size, keywords, top_k, enforce_keywords,
                    window_size, initial_count, filter_stages
                ],
                outputs=[
                    search_status, search_results_preview, 
                    results_accordion, analysis_section, search_result_state
                ]
            )
            
            analyze_btn.click(
                self._perform_analysis,
                inputs=[
                    question, search_result_state, model_choice, 
                    temperature, openai_api_key
                ],
                outputs=[analysis_output]
            )
            
            find_similar_btn.click(
                self._find_similar_words,
                inputs=[keyword_input],
                outputs=[similar_words_output]
            )
        
        return app
    
    def _update_mode_options(self, mode: str) -> Tuple[gr.Group, gr.Group]:
        """Show/hide mode-specific options"""
        timewindow_visible = mode == "Zeitfenster"
        agent_visible = mode == "Agent"
        
        return (
            gr.update(visible=timewindow_visible),
            gr.update(visible=agent_visible)
        )
    
    def _perform_search(self, 
                       mode: str,
                       content_description: str,
                       year_start: int,
                       year_end: int,
                       chunk_size: int,
                       keywords: str,
                       top_k: int,
                       enforce_keywords: bool,
                       window_size: int,
                       initial_count: int,
                       filter_stages: List[str]) -> Tuple[str, str, gr.Accordion, gr.Group, Any]:
        """Execute search based on selected mode"""
        
        if not content_description.strip():
            return (
                "❌ Bitte geben Sie eine Inhaltsbeschreibung ein.",
                "",
                gr.update(open=False),
                gr.update(visible=False),
                None
            )
        
        try:
            # Create search config
            config = SearchConfig(
                content_description=content_description,
                year_range=(year_start, year_end),
                chunk_size=chunk_size,
                keywords=keywords.strip() if keywords else None,
                top_k=top_k,
                enforce_keywords=enforce_keywords
            )
            
            # Execute search
            result = self.search_service.execute_search(
                mode=mode.lower(),
                config=config,
                window_size=window_size if mode == "Zeitfenster" else None,
                initial_count=initial_count if mode == "Agent" else None,
                filter_stages=[int(x) for x in filter_stages] if mode == "Agent" else None
            )
            
            # Format results
            status = f"✅ {result.chunk_count} Texte gefunden ({result.metadata['search_time']:.2f}s)"
            preview = self._format_search_results(result)
            
            return (
                status,
                preview,
                gr.update(open=True),  # Open results accordion
                gr.update(visible=True),  # Show analysis section
                result  # Store in state
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return (
                f"❌ Fehler: {str(e)}",
                "",
                gr.update(open=False),
                gr.update(visible=False),
                None
            )
    
    def _perform_analysis(self,
                         question: str,
                         search_result: Any,
                         model: str,
                         temperature: float,
                         openai_api_key: str) -> str:
        """Analyze search results with LLM"""
        
        if not question.strip():
            return "❌ Bitte stellen Sie eine Frage."
        
        if not search_result:
            return "❌ Bitte führen Sie zuerst eine Suche durch."
        
        try:
            result = self.search_service.analyze_results(
                question=question,
                search_result=search_result,
                model=model,
                temperature=temperature,
                openai_api_key=openai_api_key if model.startswith("gpt") else None
            )
            
            return f"### Antwort\n\n{result.answer}\n\n---\n*Modell: {result.model}*"
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return f"❌ Fehler bei der Analyse: {str(e)}"
    
    def _find_similar_words(self, word: str) -> str:
        """Find similar words"""
        if not word.strip():
            return "Bitte geben Sie ein Wort ein."
        
        try:
            similar = self.search_service.find_similar_words(word.strip())
            
            if not similar:
                return f"Keine ähnlichen Wörter für '{word}' gefunden."
            
            result = f"### Ähnliche Wörter zu '{word}':\n\n"
            for item in similar[:10]:
                result += f"- **{item['word']}** (Ähnlichkeit: {item['similarity']:.3f})\n"
            
            return result
            
        except Exception as e:
            return f"Fehler: {str(e)}"
    
    def _format_search_results(self, result: Any) -> str:
        """Format search results for preview"""
        chunks = result.chunks[:5]  # Show first 5
        
        if not chunks:
            return "Keine Ergebnisse gefunden."
        
        output = f"### Top {len(chunks)} von {result.chunk_count} Ergebnissen\n\n"
        
        for i, (doc, score) in enumerate(chunks):
            metadata = doc.metadata
            output += f"**{i+1}. {metadata.get('Artikeltitel', 'Kein Titel')}**\n"
            output += f"*{metadata.get('Datum', 'Unbekannt')} - Relevanz: {score:.3f}*\n"
            output += f"{doc.page_content[:200]}...\n\n"
        
        if result.chunk_count > 5:
            output += f"*... und {result.chunk_count - 5} weitere Ergebnisse*"
        
        return output
    
    def _get_help_text(self) -> str:
        """Get help text"""
        return """
        ## Suchmodi
        
        ### Standard
        Einfache Ähnlichkeitssuche - schnell und direkt.
        
        ### Zeitfenster
        Sucht in definierten Zeitabschnitten für bessere zeitliche Abdeckung.
        
        ### Agent
        Mehrstufige Suche mit KI-Bewertung für präzisere Ergebnisse.
        
        ## Tipps
        
        1. Beginnen Sie mit einer klaren Inhaltsbeschreibung
        2. Nutzen Sie Schlagwörter für präzisere Filterung
        3. Der Agent-Modus ist langsamer aber genauer
        4. Speichern Sie Ihren OpenAI API Key in der Umgebung
        """


def create_app() -> gr.Blocks:
    """Create and launch the application"""
    ui = UnifiedSearchUI()
    return ui.create_interface()


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)