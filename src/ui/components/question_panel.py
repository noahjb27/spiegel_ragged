# src/ui/components/question_panel.py 
import gradio as gr
from typing import Dict, Any

from src.config import settings

def create_question_panel() -> Dict[str, Any]:
    """
    Returns:
        Dictionary of UI components
    """
    with gr.Group(elem_classes=["form-container"]):
        gr.HTML("<h3 style='margin-top: 0; color: var(--text-primary);'>📊 Analyse</h3>")
   
        
        # 1. QUELLENAUSWAHL with chunk selection
        with gr.Accordion("1. Quellenauswahl", open=True):
            gr.Markdown("""
            ### Quellenauswahl für Analyse
            
            Wählen Sie aus den gefundenen Quellen diejenigen aus, die Sie für die Analyse verwenden möchten.
            Alle Quellen sind standardmäßig vorausgewählt.
            """)
            
            # Chunk selection display area - will be populated dynamically
            chunk_selection_area = gr.HTML(
                value="<p><em>Quellen werden nach der Heuristik hier angezeigt...</em></p>",
                label="Verfügbare Quellen"
            )
            
            # Selection summary and transfer button
            with gr.Row():
                selection_summary = gr.Markdown(
                    value="**Keine Quellen verfügbar**",
                    elem_id="selection_summary"
                )
                
                transfer_selection_btn = gr.Button(
                    "Auswahl in Analyse übertragen",
                    variant="primary",
                    visible=False,
                    elem_id="transfer_btn"
                )
            
            # Hidden state for selected chunks
            selected_chunks_state = gr.State([])
            chunks_transferred_state = gr.State(False)
        
        # 2. USER-PROMPT FORMULIEREN
        with gr.Accordion("2. User-Prompt formulieren", open=False) as user_prompt_accordion:
            gr.Markdown("""
            ### Forschungsfrage formulieren
            
            Formulieren Sie hier Ihre konkrete Frage an die ausgewählten Quellen.
            """)
            
            user_prompt = gr.Textbox(
                label="User-Prompt (was Sie über die Quellen wissen möchten)",
                placeholder="Beispiel: Wie wurde die Berliner Mauer in den westdeutschen Medien dargestellt?",
                lines=3,
                info="Formulieren Sie Ihre Forschungsfrage, die anhand der ausgewählten Texte beantwortet werden soll."
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
            
            **Hilfestellung für eigene Prompts:**
            - Definieren Sie die Rolle (z.B. "Historiker", "Medienanalyst")
            - Geben Sie methodische Anweisungen (Quellentreue, Belege, Struktur)
            - Spezifizieren Sie das gewünschte Antwortformat
            - Betonen Sie wissenschaftliche Standards
            """) # HIER NOCH ETWAS ZU METAFRAGEN
            
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
        
        # UPDATED: Analyse starten button
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
    
    # Event handlers for chunk selection
    def update_chunk_selection_display(retrieved_chunks):
        """Update the chunk selection display with checkboxes."""
        if not retrieved_chunks or not retrieved_chunks.get('chunks'):
            return (
                "<p><em>Keine Quellen verfügbar. Führen Sie zuerst eine Heuristik durch.</em></p>",
                "**Keine Quellen verfügbar**",
                gr.update(visible=False),
                []
            )
        
        chunks = retrieved_chunks.get('chunks', [])
        
        # Create HTML with checkboxes for each chunk
        html_content = "<div style='max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; border-radius: 8px;'>"
        
        for i, chunk in enumerate(chunks):
            chunk_id = i + 1
            metadata = chunk.get('metadata', {})
            title = metadata.get('Artikeltitel', 'Kein Titel')
            date = metadata.get('Datum', 'Unbekannt')
            relevance = chunk.get('relevance_score', 0.0)
            
            # Truncate content for preview
            content_preview = chunk.get('content', '')[:200]
            if len(chunk.get('content', '')) > 200:
                content_preview += '...'
            
            html_content += f"""
            <div style='margin-bottom: 15px; padding: 10px; border: 1px solid #eee; border-radius: 5px; background-color: #fafafa;'>
                <div style='margin-bottom: 8px;'>
                    <input type='checkbox' id='chunk_{chunk_id}' checked onchange='updateSelection()' style='margin-right: 8px;'>
                    <label for='chunk_{chunk_id}' style='font-weight: bold; cursor: pointer;'>
                        {chunk_id}. {title} ({date})
                    </label>
                    <span style='margin-left: 10px; color: #666; font-size: 0.9em;'>Relevanz: {relevance:.3f}</span>
                </div>
                <div style='margin-left: 24px; color: #555; font-size: 0.9em; line-height: 1.4;'>
                    {content_preview}
                </div>
            </div>
            """
        
        html_content += """
        </div>
        
        <script>
        function updateSelection() {
            // This would need to be handled by Gradio's event system
            console.log('Selection updated');
        }
        </script>
        """
        
        # Initialize with all chunks selected
        all_chunk_ids = list(range(1, len(chunks) + 1))
        summary_text = f"**Ausgewählt**: {len(chunks)} von {len(chunks)} Quellen"
        
        return (
            html_content,
            summary_text,
            gr.update(visible=True),
            all_chunk_ids
        )
    
    def transfer_chunks_to_analysis(selected_chunks, chunks_data):
        """Transfer selected chunks to analysis section."""
        if not selected_chunks:
            return (
                gr.update(visible=False),
                "❌ Keine Quellen ausgewählt",
                False,
                gr.update(open=True),
                gr.update(open=False),
                gr.update(open=False),
                gr.update(open=False)
            )
        
        # Update UI to show analysis is ready
        return (
            gr.update(visible=True),
            f"✅ {len(selected_chunks)} Quellen in die Analyse übertragen",
            True,
            gr.update(open=False),  # Close source selection
            gr.update(open=True),   # Open user prompt
            gr.update(open=False),  # Keep LLM selection closed
            gr.update(open=False)   # Keep system prompt closed
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
    
    # Transfer button click
    transfer_selection_btn.click(
        transfer_chunks_to_analysis,
        inputs=[selected_chunks_state, selected_chunks_state],  # Placeholder for now
        outputs=[
            analyze_btn,
            analysis_status,
            chunks_transferred_state,
            user_prompt_accordion,
            llm_selection_accordion,
            system_prompt_accordion,
            analysis_status
        ]
    )
    
    # Define all components to be returned
    components = {
        # NEW: Source selection components
        "chunk_selection_area": chunk_selection_area,
        "selection_summary": selection_summary,
        "transfer_selection_btn": transfer_selection_btn,
        "selected_chunks_state": selected_chunks_state,
        "chunks_transferred_state": chunks_transferred_state,
        
        # UPDATED: Analysis components with new names
        "user_prompt": user_prompt,  # UPDATED from "question"
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
        
        # Helper function for external use
        "update_chunk_selection_display": update_chunk_selection_display,
        
        # DEPRECATED: Keep for backward compatibility
        "question": user_prompt,  # Alias
        "chunk_selection_mode": gr.State("all"),  # Dummy for compatibility
        "chunk_selection_file": gr.State(None),  # Dummy for compatibility
        "upload_status": gr.State(""),  # Dummy for compatibility
        "upload_preview": gr.State(""),  # Dummy for compatibility
        "manual_chunk_ids": gr.State(""),  # Dummy for compatibility
        "manual_status": gr.State(""),  # Dummy for compatibility
        "selection_summary_old": selection_summary,  # Alias
        "custom_system_prompt": system_prompt_text  # Alias for backward compatibility
    }
    
    return components