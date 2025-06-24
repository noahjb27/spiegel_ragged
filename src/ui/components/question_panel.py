# src/ui/components/question_panel.py - FIXED: Dark theme CSS integration
import gradio as gr
from typing import Dict, Any

from src.config import settings

def create_question_panel() -> Dict[str, Any]:
    """Create updated question panel that shows transferred chunks with proper dark theme."""
    
    with gr.Group(elem_classes=["form-container"]):
        gr.HTML("<h3 style='margin-top: 0; color: var(--text-primary);'>📊 Analyse</h3>")
   
        # 1. ÜBERTRAGENE QUELLEN - FIXED: Better dark theme integration
        with gr.Accordion("1. Übertragene Quellen", open=True):
            gr.Markdown("""
            ### Aus der Heuristik übertragene Quellen
            
            Diese Texte wurden aus der Heuristik-Phase zur Analyse übertragen.
            Um die Auswahl zu ändern, kehren Sie zur Heuristik zurück.
            """)
            
            # FIXED: Display area with proper dark theme styling
            transferred_chunks_display = gr.HTML(
                value="""<div class='info-message'>
                <p><em>Noch keine Quellen übertragen. Führen Sie zuerst eine Heuristik durch und übertragen Sie Quellen.</em></p>
                </div>""",
                label="Übertragene Quellen"
            )
            
            # Summary with proper styling
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
                    label="System-Prompt Vorlage",
                    info="Grundlegende Vorlagen für wissenschaftliche Analyse"
                )
                
                reset_system_prompt_btn = gr.Button("Auf Vorlage zurücksetzen", size="sm")
            
            system_prompt_text = gr.Textbox(
                label="System-Prompt (bearbeitbar)",
                value=settings.SYSTEM_PROMPTS["default"],
                lines=8,
                info="Bearbeiten Sie den System-Prompt für eine präzise wissenschaftliche Analyse."
            )
        
        # ANALYSE STARTEN button
        analyze_btn = gr.Button("Analyse starten", variant="primary", visible=False)
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
        FIXED: Update the display of transferred chunks with proper dark theme CSS.
        """
        if not transferred_chunks:
            return (
                """<div class="info-message">
                <p><em>Noch keine Quellen übertragen. Führen Sie zuerst eine Heuristik durch und übertragen Sie Quellen.</em></p>
                </div>""",
                "**Keine Quellen übertragen**",
                gr.update(visible=False)  # Hide analyze button
            )
        
        # FIXED: Create HTML display with proper dark theme CSS
        html_content = f"""
        <div class="results-container" style="max-height: 60vh; overflow-y: auto;">
            <div class="success-message" style="margin-bottom: 20px;">
                <h4 style="color: var(--text-primary); margin-top: 0;">
                    ✅ Zur Analyse übertragene Texte ({len(transferred_chunks)})
                </h4>
                <p style="color: var(--text-secondary);">
                    Diese Texte wurden aus der Heuristik übertragen und werden für die Analyse verwendet.
                </p>
            </div>
        """
        
        for i, chunk in enumerate(transferred_chunks):
            chunk_id = chunk.get('transferred_id', i + 1)
            metadata = chunk.get('metadata', {})
            title = metadata.get('Artikeltitel', 'Kein Titel')
            date = metadata.get('Datum', 'Unbekannt')
            relevance = chunk.get('relevance_score', 0.0)
            url = metadata.get('URL', '')
            
            # Get content with reasonable preview length
            content = chunk.get('content', '')
            content_preview = content[:400] + '...' if len(content) > 400 else content
            
            # Create URL link if available
            title_display = title
            if url and url != 'Keine URL':
                title_display = f'<a href="{url}" target="_blank" style="color: var(--brand-primary); text-decoration: none;">{title} 🔗</a>'
            
            # FIXED: Use proper CSS classes and variables for dark theme
            html_content += f"""
            <div class="evaluation-card" style="margin-bottom: 15px;">
                <div style="margin-bottom: 10px;">
                    <div style="color: var(--text-primary); font-weight: 600; font-size: 16px; margin-bottom: 6px;">
                        {chunk_id}. {title_display}
                    </div>
                    <div style="color: var(--text-secondary); font-size: 14px; margin-bottom: 10px;">
                        <strong>Datum:</strong> {date} | <strong>Relevanz:</strong> {relevance:.3f}
                    </div>
                </div>
                
                <details style="margin-top: 10px;">
                    <summary style="
                        color: var(--text-primary); 
                        font-weight: 500; 
                        cursor: pointer; 
                        padding: 5px 0;
                        border-bottom: 1px solid var(--border-primary);
                    ">
                        📄 Textvorschau anzeigen
                    </summary>
                    <div style="
                        border-left: 3px solid var(--brand-secondary); 
                        padding: 12px; 
                        background: var(--bg-primary); 
                        border-radius: 4px; 
                        margin-top: 10px;
                        color: var(--text-secondary);
                        line-height: 1.6;
                        white-space: pre-wrap;
                        max-height: 300px;
                        overflow-y: auto;
                    ">
                        {content_preview}
                    </div>
                </details>
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
    system_prompt_template.change(
        load_system_prompt_template,
        inputs=[system_prompt_template],
        outputs=[system_prompt_text]
    )
    
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
        "chunk_selection_mode": gr.State("transferred"),
        "selected_chunks_state": transferred_chunks_state,
    }
    
    return components