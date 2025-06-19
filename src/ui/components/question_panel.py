# src/ui/components/question_panel.py - Enhanced with chunk selection capability
"""
Enhanced question panel component with ability to load specific chunks by ID.
"""
import gradio as gr
from typing import Dict, Any

from src.config import settings

def create_question_panel() -> Dict[str, Any]:
    """
    Create the enhanced question panel UI components with chunk selection functionality.
    
    Returns:
        Dictionary of UI components
    """
    with gr.Group():
        gr.Markdown("## Frage stellen")
        
        question = gr.Textbox(
            label="Frage (was Sie über die gefundenen Inhalte wissen möchten)",
            placeholder="Beispiel: Wie wurde die Berliner Mauer in den westdeutschen Medien dargestellt?",
            lines=2,
            info="Formulieren Sie Ihre Frage, die anhand der gefundenen Texte beantwortet werden soll."
        )
        
        # ENHANCED: Chunk selection options
        with gr.Accordion("Quellenauswahl", open=False):
            gr.Markdown("""
            ### Quellenauswahl für Analyse
            
            Sie können entweder alle gefundenen Quellen verwenden oder eine spezielle Auswahl treffen:
            
            - **Alle verwenden**: Nutzt alle bei der Suche gefundenen Texte
            - **Auswahl hochladen**: Laden Sie eine CSV/JSON-Datei mit den gewünschten Chunk-IDs hoch
            - **Manual IDs**: Geben Sie kommagetrennte Chunk-IDs direkt ein
            """)
            
            chunk_selection_mode = gr.Radio(
                choices=[
                    ("Alle gefundenen Quellen verwenden", "all"),
                    ("Auswahl aus Datei laden", "upload"),
                    ("Chunk-IDs manuell eingeben", "manual")
                ],
                value="all",
                label="Quellenauswahl-Modus"
            )
            
            # File upload for chunk selection
            with gr.Group(visible=False) as upload_group:
                chunk_selection_file = gr.File(
                    label="Chunk-Auswahl-Datei",
                    file_types=[".csv", ".json"],
                    info="Laden Sie eine CSV- oder JSON-Datei mit den gewünschten Chunk-IDs hoch."
                )
                
                upload_status = gr.Markdown("", visible=False)
                
                # Preview of uploaded selection
                upload_preview = gr.Markdown("", visible=False)
            
            # Manual chunk ID input
            with gr.Group(visible=False) as manual_group:
                manual_chunk_ids = gr.Textbox(
                    label="Chunk-IDs (kommagetrennt)",
                    placeholder="1,3,5,7,9",
                    lines=2,
                    info="Geben Sie die IDs der Texte ein, die Sie für die Analyse verwenden möchten."
                )
                
                manual_status = gr.Markdown("", visible=False)
            
            # Current selection summary
            selection_summary = gr.Markdown("**Aktuelle Auswahl**: Alle gefundenen Quellen werden verwendet.")
        
        # Model selection settings
        with gr.Accordion("LLM-Einstellungen", open=False):
            gr.Markdown("""
            ### Modellauswahl
            
            Sie können zwischen verschiedenen LLM-Modellen wählen:
            - **HU-LLM 1**: Lokales Modell (HU-Netzwerk erforderlich)
            - **HU-LLM 3**: Lokales Modell (HU-Netzwerk erforderlich)  
            - **DeepSeek R1 32B**: Fortschrittliches Reasoning-Modell via Ollama (HU-Netzwerk erforderlich)
            - **OpenAI GPT-4o**: Leistungsstärkstes OpenAI-Modell
            - **Google Gemini 2.5 Pro**: Googles intelligentestes Modell mit großem Kontextfenster
            
            """)
            
            with gr.Row():
                model_selection = gr.Radio(
                    choices=["hu-llm1", "hu-llm3", "deepseek-r1", "openai-gpt4o", "gemini-pro"],
                    value="hu-llm3",
                    label="LLM-Modell",
                    info="Wählen Sie das zu verwendende Sprachmodell."
                )
            
            # System prompt template selection and editing
            gr.Markdown("""
            ### System Prompt
            
            Wählen Sie eine Vorlage und bearbeiten Sie sie nach Ihren Bedürfnissen. Der System Prompt steuert, 
            wie das LLM die Analyse durchführt.
            """)
            
            with gr.Row():
                system_prompt_template = gr.Dropdown(
                    choices=list(settings.SYSTEM_PROMPTS.keys()),
                    value="default",
                    label="System Prompt Vorlage",
                    info="Wählen Sie eine Vorlage als Ausgangspunkt"
                )
                
                reset_system_prompt_btn = gr.Button("Auf Vorlage zurücksetzen", size="sm")
            
            # Editable system prompt text area - initialized with default template
            system_prompt_text = gr.Textbox(
                label="System Prompt (bearbeitbar)",
                value=settings.SYSTEM_PROMPTS["default"],
                lines=8,
                info="Bearbeiten Sie den System Prompt nach Ihren Bedürfnissen. Dieser Text wird an das LLM gesendet."
            )
            
            # Add temperature and max tokens
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                    label="Temperatur",
                    info="Bestimmt die Wahrscheinlichkeitsverteilung, aus der Tokens ausgewählt werden – höher bedeutet kreativere, aber potenziell weniger kohärente Texte."
                )
        
        analyze_btn = gr.Button("Frage beantworten", variant="primary")
        
        # Hidden state for selected chunks
        selected_chunks_state = gr.State(None)
    
    # Event handlers for system prompt template management
    def load_system_prompt_template(template_name: str) -> str:
        """Load the selected template into the text area."""
        return settings.SYSTEM_PROMPTS.get(template_name, settings.SYSTEM_PROMPTS["default"])
    
    def reset_to_template(template_name: str) -> str:
        """Reset the text area to the selected template."""
        return settings.SYSTEM_PROMPTS.get(template_name, settings.SYSTEM_PROMPTS["default"])
    
    # Event handlers for chunk selection
    def toggle_chunk_selection_ui(mode: str):
        """Show/hide UI elements based on selection mode."""
        if mode == "upload":
            return gr.update(visible=True), gr.update(visible=False), "**Aktuelle Auswahl**: Wird aus hochgeladener Datei geladen."
        elif mode == "manual":
            return gr.update(visible=False), gr.update(visible=True), "**Aktuelle Auswahl**: Manuelle Chunk-ID-Eingabe."
        else:
            return gr.update(visible=False), gr.update(visible=False), "**Aktuelle Auswahl**: Alle gefundenen Quellen werden verwendet."
    
    def process_uploaded_chunk_selection(file_obj):
        """Process uploaded chunk selection file."""
        if not file_obj:
            return "Keine Datei hochgeladen.", "", None
        
        try:
            import json
            import csv
            import os
            
            file_path = file_obj.name
            file_extension = os.path.splitext(file_path)[1].lower()
            
            chunk_ids = []
            
            if file_extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Try different JSON structures
                if isinstance(data, list):
                    # Simple list of IDs
                    chunk_ids = [int(x) for x in data if str(x).isdigit()]
                elif isinstance(data, dict):
                    # Dictionary with chunk_ids key
                    if 'chunk_ids' in data:
                        chunk_ids = [int(x) for x in data['chunk_ids'] if str(x).isdigit()]
                    elif 'chunks' in data:
                        # Extract chunk_id from chunks list
                        for chunk in data['chunks']:
                            if isinstance(chunk, dict) and 'chunk_id' in chunk:
                                chunk_ids.append(int(chunk['chunk_id']))
                            elif isinstance(chunk, dict) and 'id' in chunk:
                                chunk_ids.append(int(chunk['id']))
            
            elif file_extension == '.csv':
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Try to detect if first column contains chunk IDs
                    csv_reader = csv.reader(f)
                    headers = next(csv_reader, None)
                    
                    # Look for chunk_id column
                    chunk_id_col = None
                    if headers:
                        for i, header in enumerate(headers):
                            if 'chunk_id' in header.lower() or 'id' in header.lower():
                                chunk_id_col = i
                                break
                    
                    # If no specific column found, use first column
                    if chunk_id_col is None:
                        chunk_id_col = 0
                    
                    # Read chunk IDs
                    f.seek(0)
                    csv_reader = csv.reader(f)
                    if headers:
                        next(csv_reader)  # Skip header
                    
                    for row in csv_reader:
                        if len(row) > chunk_id_col and row[chunk_id_col].strip().isdigit():
                            chunk_ids.append(int(row[chunk_id_col]))
            
            if not chunk_ids:
                return "❌ Keine gültigen Chunk-IDs in der Datei gefunden.", "", None
            
            # Remove duplicates and sort
            chunk_ids = sorted(list(set(chunk_ids)))
            
            status_msg = f"✅ {len(chunk_ids)} Chunk-IDs erfolgreich geladen."
            preview_msg = f"**Geladene Chunk-IDs**: {', '.join(map(str, chunk_ids[:20]))}" + ("..." if len(chunk_ids) > 20 else "")
            
            return status_msg, preview_msg, chunk_ids
            
        except Exception as e:
            return f"❌ Fehler beim Verarbeiten der Datei: {str(e)}", "", None
    
    def process_manual_chunk_ids(ids_text: str):
        """Process manually entered chunk IDs."""
        if not ids_text.strip():
            return "Keine IDs eingegeben.", None
        
        try:
            # Parse comma-separated IDs
            ids_text = ids_text.strip()
            chunk_ids = []
            
            for part in ids_text.split(','):
                part = part.strip()
                if part.isdigit():
                    chunk_ids.append(int(part))
                elif '-' in part and len(part.split('-')) == 2:
                    # Handle ranges like "1-5"
                    start, end = part.split('-')
                    if start.strip().isdigit() and end.strip().isdigit():
                        start_id, end_id = int(start.strip()), int(end.strip())
                        chunk_ids.extend(range(start_id, end_id + 1))
            
            if not chunk_ids:
                return "❌ Keine gültigen Chunk-IDs gefunden.", None
            
            # Remove duplicates and sort
            chunk_ids = sorted(list(set(chunk_ids)))
            
            status_msg = f"✅ {len(chunk_ids)} Chunk-IDs erfolgreich verarbeitet: {', '.join(map(str, chunk_ids))}"
            return status_msg, chunk_ids
            
        except Exception as e:
            return f"❌ Fehler beim Verarbeiten der IDs: {str(e)}", None
    
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
    
    # Chunk selection mode changes
    chunk_selection_mode.change(
        toggle_chunk_selection_ui,
        inputs=[chunk_selection_mode],
        outputs=[upload_group, manual_group, selection_summary]
    )
    
    # File upload processing
    chunk_selection_file.change(
        process_uploaded_chunk_selection,
        inputs=[chunk_selection_file],
        outputs=[upload_status, upload_preview, selected_chunks_state]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[upload_status]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[upload_preview]
    )
    
    # Manual ID processing
    manual_chunk_ids.change(
        process_manual_chunk_ids,
        inputs=[manual_chunk_ids],
        outputs=[manual_status, selected_chunks_state]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[manual_status]
    )
    
    # Define all components to be returned
    components = {
        "question": question,
        "model_selection": model_selection,
        "system_prompt_template": system_prompt_template,
        "system_prompt_text": system_prompt_text,
        "reset_system_prompt_btn": reset_system_prompt_btn,
        "temperature": temperature,
        "analyze_btn": analyze_btn,
        
        # ENHANCED: Chunk selection components
        "chunk_selection_mode": chunk_selection_mode,
        "chunk_selection_file": chunk_selection_file,
        "upload_status": upload_status,
        "upload_preview": upload_preview,
        "manual_chunk_ids": manual_chunk_ids,
        "manual_status": manual_status,
        "selection_summary": selection_summary,
        "selected_chunks_state": selected_chunks_state,
        
        # Keep these for backward compatibility but mark as deprecated
        "custom_system_prompt": system_prompt_text  # Alias for backward compatibility
    }
    
    return components