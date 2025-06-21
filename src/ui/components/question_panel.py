# src/ui/components/question_panel.py - Updated to show transferred chunks
import gradio as gr
from typing import Dict, Any

from src.config import settings

def create_question_panel() -> Dict[str, Any]:
    """
    Create updated question panel that shows transferred chunks instead of allowing selection.
    
    Returns:
        Dictionary of UI components
    """
    with gr.Group(elem_classes=["form-container"]):
        gr.HTML("<h3 style='margin-top: 0; color: var(--text-primary);'>📊 Analyse</h3>")
   
        # 1. ÜBERTRAGENE QUELLEN (Read-only display)
        with gr.Accordion("1. Übertragene Quellen", open=True):
            gr.Markdown("""
            ### Aus der Heuristik übertragene Quellen
            
            Diese Texte wurden aus der Heuristik-Phase zur Analyse übertragen.
            Um die Auswahl zu ändern, kehren Sie zur Heuristik zurück.
            """)
            
            # Display area for transferred chunks (read-only)
            transferred_chunks_display = gr.HTML(
                value="<p><em>Noch keine Quellen übertragen. Führen Sie zuerst eine Heuristik durch und übertragen Sie Quellen.</em></p>",
                label="Übertragene Quellen"
            )
            
            # Summary of transferred chunks
            transferred_summary = gr.Markdown(
                value="**Keine Quellen übertragen**",
                elem_id="transferred_summary"
            )
            
            # Hidden state for transferred chunks
            transferred_chunks_state = gr.State([])
        
        # 2. USER-PROMPT FORMULIEREN
        with gr.Accordion("2. User-Prompt formulieren", open=False) as user_prompt_accordion:
            gr.Markdown("""
            ### Forschungsfrage formulieren
            
            Formulieren Sie hier Ihre konkrete Frage an die übertragenen Quellen.
            """)
            
            user_prompt = gr.Textbox(
                label="User-Prompt (was Sie über die Quellen wissen möchten)",
                placeholder="Beispiel: Wie wurde die Berliner Mauer in den westdeutschen Medien dargestellt?",
                lines=3,
                info="Formulieren Sie Ihre Forschungsfrage, die anhand der übertragenen Texte beantwortet werden soll."
            )
        
        # 3. LLM-AUSWÄHLEN
        with gr.Accordion("3. LLM-Auswählen", open=False) as llm_selection_accordion:
            gr.Markdown("""
            ### Modellauswahl
            
            Wählen Sie das Sprachmodell für die Analyse:
            """)
            
            model_selection = gr.Radio(
                choices=["hu-llm1", "hu-llm3", "deepseek-r1", "openai-gpt4o", "gemini-pro"],
                value="hu-llm3",
                label="LLM-Modell",
                info="Wählen Sie das zu verwendende Sprachmodell."
            )
            
            # Temperature control
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.1,
                label="Temperatur",
                info="Bestimmt den Determinismus der Antwortgenerierung. Niedrigere Werte = konsistentere, fokussiertere Antworten."
            )
        
        # 4. SYSTEM-PROMPT
        with gr.Accordion("4. System-Prompt", open=False) as system_prompt_accordion:
            gr.Markdown("""
            ### System-Prompt konfigurieren
            
            Der System-Prompt steuert, wie das LLM die Analyse durchführt. 
            Fokussieren Sie auf akademische Präzision, Quellentreue und wissenschaftliche Methodik.
            """)
            
            with gr.Row():
                system_prompt_template = gr.Dropdown(
                    choices=["default"],
                    value="default",
                    label="System Prompt Vorlage",
                    info="Grundlegende Vorlagen für wissenschaftliche Analyse"
                )
                
                reset_system_prompt_btn = gr.Button("Auf Vorlage zurücksetzen", size="sm")
            
            # Editable system prompt text area
            system_prompt_text = gr.Textbox(
                label="System Prompt (bearbeitbar)",
                value=settings.SYSTEM_PROMPTS["default"],
                lines=8,
                info="Bearbeiten Sie den System Prompt für eine präzise wissenschaftliche Analyse."
            )
        
        # ANALYSE STARTEN button
        analyze_btn = gr.Button("Analyse starten", variant="primary", visible=False)
        
        # Analysis status
        analysis_status = gr.Markdown("", visible=False)
    
    # Event handlers for system prompt template management
    def load_system_prompt_template(template_name: str) -> str:
        """Load the selected template into the text area."""
        return settings.SYSTEM_PROMPTS.get(template_name, settings.SYSTEM_PROMPTS["default"])
    
    def reset_to_template(template_name: str) -> str:
        """Reset the text area to the selected template."""
        return settings.SYSTEM_PROMPTS.get(template_name, settings.SYSTEM_PROMPTS["default"])
    
    def update_transferred_chunks_display(transferred_chunks: list) -> tuple:
        """
        Update the display of transferred chunks.
        
        Args:
            transferred_chunks: List of transferred chunk data
            
        Returns:
            Tuple of (display_html, summary_text, analyze_btn_visibility)
        """
        if not transferred_chunks:
            return (
                "<p><em>Noch keine Quellen übertragen. Führen Sie zuerst eine Heuristik durch und übertragen Sie Quellen.</em></p>",
                "**Keine Quellen übertragen**",
                gr.update(visible=False)  # Hide analyze button
            )
        
        # Create HTML display for transferred chunks
        html_content = f"""
        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 8px; background-color: #f9f9f9;">
            <h4 style="margin-top: 0; color: #2c3e50;">Zur Analyse übertragene Texte ({len(transferred_chunks)})</h4>
        """
        
        for i, chunk in enumerate(transferred_chunks):
            chunk_id = chunk.get('transferred_id', i + 1)
            metadata = chunk.get('metadata', {})
            title = metadata.get('Artikeltitel', 'Kein Titel')
            date = metadata.get('Datum', 'Unbekannt')
            relevance = chunk.get('relevance_score', 0.0)
            url = metadata.get('URL', '')
            
            # Create content preview
            content = chunk.get('content', '')
            content_preview = content[:200] + '...' if len(content) > 200 else content
            
            # Create URL link if available
            title_display = title
            if url and url != 'Keine URL':
                title_display = f'<a href="{url}" target="_blank" style="color: #d75425; text-decoration: none;">{title}</a>'
            
            html_content += f"""
            <div style="margin-bottom: 15px; padding: 12px; border: 1px solid #e0e0e0; border-radius: 6px; background-color: white;">
                <div style="font-weight: bold; color: #2c3e50; margin-bottom: 5px;">
                    {chunk_id}. {title_display}
                </div>
                <div style="font-size: 13px; color: #666; margin-bottom: 8px;">
                    <strong>Datum:</strong> {date} | <strong>Relevanz:</strong> {relevance:.3f}
                </div>
                <div style="font-size: 13px; color: #555; line-height: 1.4; border-left: 3px solid #b2b069; padding-left: 10px;">
                    {content_preview}
                </div>
            </div>
            """
        
        html_content += "</div>"
        
        # Create summary
        summary_text = f"**Übertragene Quellen**: {len(transferred_chunks)} Texte bereit für die Analyse"
        
        return (
            html_content,
            summary_text,
            gr.update(visible=True)  # Show analyze button
        )
    
    # Connect event handlers
    
    # Template dropdown to text area
    system_prompt_template.change(
        load_system_prompt_template,
        inputs=[system_prompt_template],
        outputs=[system_prompt_text]
    )
    
    # Reset button
    reset_system_prompt_btn.click(
        reset_to_template,
        inputs=[system_prompt_template],
        outputs=[system_prompt_text]
    )
    
    # Define all components to be returned
    components = {
        # NEW: Transferred chunks components
        "transferred_chunks_display": transferred_chunks_display,
        "transferred_summary": transferred_summary,
        "transferred_chunks_state": transferred_chunks_state,
        "update_transferred_chunks_display": update_transferred_chunks_display,
        
        # Analysis components
        "user_prompt": user_prompt,
        "model_selection": model_selection,
        "system_prompt_template": system_prompt_template,
        "system_prompt_text": system_prompt_text,
        "reset_system_prompt_btn": reset_system_prompt_btn,
        "temperature": temperature,
        "analyze_btn": analyze_btn,
        "analysis_status": analysis_status,
        
        # Accordion references for dynamic control
        "user_prompt_accordion": user_prompt_accordion,
        "llm_selection_accordion": llm_selection_accordion,
        "system_prompt_accordion": system_prompt_accordion,
        
        # BACKWARD COMPATIBILITY: Aliases for existing code
        "question": user_prompt,
        "chunk_selection_mode": gr.State("transferred"),  # Always use transferred chunks
        "selected_chunks_state": transferred_chunks_state,  # Map to transferred chunks
    }
    
    return components