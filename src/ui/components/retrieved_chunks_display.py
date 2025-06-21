# src/ui/components/retrieved_chunks_display.py
"""
Component for displaying retrieved chunks with selection capabilities in the Heuristik section.
UPDATED: Now uses the centralized CSS design system with proper dark theme support.
"""
import gradio as gr
from typing import Dict, List, Any, Optional
import json

def create_retrieved_chunks_display() -> Dict[str, Any]:
    """
    Create the retrieved chunks display component with selection capabilities.
    UPDATED: Uses CSS classes from main design system.
    
    Returns:
        Dictionary of UI components for chunk display and selection
    """
    
    # UPDATED: Use form-container CSS class from main design system
    with gr.Group(elem_classes=["form-container"]):
        gr.HTML("<h3 style='margin-top: 0; color: var(--text-primary);'>📄 Gefundene Texte</h3>")
        
        # Chunk selection area with checkboxes - UPDATED: Uses main CSS
        chunks_selection_html = gr.HTML(
            value="<div class='info-message'><p><em>Führen Sie zuerst eine Heuristik durch, um Texte hier anzuzeigen...</em></p></div>",
            label="Gefundene Texte mit Auswahlmöglichkeit"
        )
        
        # Selection summary and controls - UPDATED: Improved layout
        with gr.Row():
            with gr.Column():
                selection_summary = gr.Markdown(
                    value="**Noch keine Texte verfügbar**",
                    elem_id="chunks_selection_summary"
                )
            
            with gr.Column():
                # Toggle buttons - UPDATED: Use proper button classes
                with gr.Row():
                    select_all_btn = gr.Button(
                        "✅ Alle auswählen", 
                        size="sm", 
                        visible=False,
                        elem_classes=["btn-secondary"]
                    )
                    deselect_all_btn = gr.Button(
                        "❌ Alle abwählen", 
                        size="sm", 
                        visible=False,
                        elem_classes=["btn-secondary"]
                    )
        
        # Transfer button - UPDATED: Use primary button class
        with gr.Row():
            transfer_to_analysis_btn = gr.Button(
                "🔄 Ausgewählte Quellen zur Analyse übertragen",
                variant="primary",
                visible=False,
                elem_id="transfer_chunks_btn"
            )
        
        # Transfer status - UPDATED: Will use CSS message classes
        transfer_status = gr.Markdown(value="", visible=False)
        
        # Hidden states for tracking
        available_chunks_state = gr.State([])  # All retrieved chunks
        selected_chunk_ids_state = gr.State([])  # Currently selected chunk IDs
        transferred_chunks_state = gr.State([])  # Chunks transferred to analysis
    
    # UPDATED: JavaScript with CSS class support for dark theme
    checkbox_js = """
    function updateChunkSelection() {
        const checkboxes = document.querySelectorAll('input[name="chunk_selection"]:checked');
        const selectedIds = Array.from(checkboxes).map(cb => parseInt(cb.value));
        
        // Update selection counter in UI
        const totalCheckboxes = document.querySelectorAll('input[name="chunk_selection"]').length;
        const selectedCount = selectedIds.length;
        
        // Update summary display
        const summaryElement = document.getElementById('chunks_selection_summary');
        if (summaryElement) {
            const summaryText = `**Verfügbare Texte**: ${totalCheckboxes} | **Ausgewählt**: ${selectedCount}`;
            summaryElement.innerHTML = summaryText;
        }
        
        return selectedIds;
    }
    
    function selectAllChunks() {
        const checkboxes = document.querySelectorAll('input[name="chunk_selection"]');
        checkboxes.forEach(cb => cb.checked = true);
        return updateChunkSelection();
    }
    
    function deselectAllChunks() {
        const checkboxes = document.querySelectorAll('input[name="chunk_selection"]');
        checkboxes.forEach(cb => cb.checked = false);
        return updateChunkSelection();
    }
    
    /* UPDATED: Style checkboxes for dark theme compatibility */
    function enhanceCheckboxStyling() {
        const style = document.createElement('style');
        style.textContent = `
            .chunk-checkbox-container {
                background: var(--bg-tertiary) !important;
                border: 1px solid var(--border-primary) !important;
                border-radius: 8px !important;
                padding: 15px !important;
                margin-bottom: 15px !important;
                transition: all 0.2s ease !important;
            }
            
            .chunk-checkbox-container:hover {
                border-color: var(--brand-primary) !important;
                background: var(--bg-elevated) !important;
            }
            
            .chunk-checkbox-container.selected {
                border-color: var(--brand-primary) !important;
                background: rgba(215, 84, 37, 0.1) !important;
            }
            
            .chunk-checkbox {
                accent-color: var(--brand-primary) !important;
                transform: scale(1.2) !important;
                margin-right: 10px !important;
            }
            
            .chunk-title {
                color: var(--text-primary) !important;
                font-weight: 600 !important;
                margin-bottom: 8px !important;
            }
            
            .chunk-metadata {
                color: var(--text-secondary) !important;
                font-size: 0.9em !important;
                margin-bottom: 8px !important;
            }
            
            .chunk-preview {
                color: var(--text-muted) !important;
                font-size: 0.85em !important;
                line-height: 1.4 !important;
                background: var(--bg-primary) !important;
                padding: 10px !important;
                border-radius: 4px !important;
                border-left: 3px solid var(--brand-secondary) !important;
            }
        `;
        document.head.appendChild(style);
    }
    
    // Apply styling when page loads
    document.addEventListener('DOMContentLoaded', enhanceCheckboxStyling);
    """
    
    # Add JavaScript to the component - UPDATED: Hidden but functional
    gr.HTML(f"<script>{checkbox_js}</script>", visible=False)
    
    return {
        "chunks_selection_html": chunks_selection_html,
        "selection_summary": selection_summary,
        "select_all_btn": select_all_btn,
        "deselect_all_btn": deselect_all_btn,
        "transfer_to_analysis_btn": transfer_to_analysis_btn,
        "transfer_status": transfer_status,
        "available_chunks_state": available_chunks_state,
        "selected_chunk_ids_state": selected_chunk_ids_state,
        "transferred_chunks_state": transferred_chunks_state
    }

def format_chunks_with_checkboxes(retrieved_chunks: Dict[str, Any]) -> str:
    """
    Format retrieved chunks as HTML with checkboxes for selection.
    UPDATED: Uses CSS classes from main design system for dark theme.
    
    Args:
        retrieved_chunks: Dictionary containing chunks and metadata
        
    Returns:
        HTML string with chunks and checkboxes using main CSS classes
    """
    if not retrieved_chunks or not retrieved_chunks.get('chunks'):
        return "<div class='info-message'><p><em>Keine Texte verfügbar.</em></p></div>"
    
    chunks = retrieved_chunks.get('chunks', [])
    
    # UPDATED: Generate HTML with CSS classes from main design system
    html_content = "<div style='max-height: 600px; overflow-y: auto; padding: 10px;'>"
    
    for i, chunk in enumerate(chunks, 1):
        chunk_id = i
        metadata = chunk.get('metadata', {})
        content = chunk.get('content', '')
        relevance_score = chunk.get('relevance_score', 0.0)
        
        # Get additional scores if available (for LLM-assisted search)
        vector_score = chunk.get('vector_similarity_score', relevance_score)
        llm_score = chunk.get('llm_evaluation_score', None)
        
        # Chunk title and basic info
        title = metadata.get('Artikeltitel', 'Kein Titel')
        date = metadata.get('Datum', 'Unbekannt')
        year = metadata.get('Jahrgang', 'Unbekannt')
        url = metadata.get('URL', '')
        
        # Create content preview
        content_preview = content[:300]
        if len(content) > 300:
            content_preview += '...'
        
        # UPDATED: Use CSS classes for styling
        html_content += f"""
        <div class="chunk-checkbox-container" id="chunk_{chunk_id}">
            <div style="display: flex; align-items: flex-start; gap: 12px;">
                <input 
                    type="checkbox" 
                    id="chunk_checkbox_{chunk_id}" 
                    name="chunk_selection" 
                    value="{chunk_id}" 
                    checked 
                    class="chunk-checkbox"
                    onchange="updateChunkSelection()"
                >
                <div style="flex: 1;">
                    <label for="chunk_checkbox_{chunk_id}" class="chunk-title" style="cursor: pointer;">
                        {chunk_id}. {title}
                    </label>
                    
                    <div class="chunk-metadata">
                        <strong>Datum:</strong> {date} | 
                        <strong>Jahr:</strong> {year} | 
                        <strong>Relevanz:</strong> {relevance_score:.3f}
        """
        
        # Add dual scores if available (LLM-assisted search)
        if llm_score is not None:
            html_content += f""" | 
                        <strong>Vector:</strong> {vector_score:.3f} | 
                        <strong>LLM:</strong> {llm_score:.3f}"""
        
        # Add URL if available
        if url and url != 'Keine URL':
            html_content += f""" | 
                        <a href="{url}" target="_blank" style="color: var(--brand-primary); text-decoration: none;">🔗 Artikel</a>"""
        
        html_content += """
                    </div>
                    
                    <div class="chunk-preview">
        """
        
        # Add evaluation text if available (LLM-assisted search)
        evaluation_text = metadata.get('evaluation_text', '')
        if evaluation_text:
            html_content += f"""
                        <div style="margin-bottom: 10px; padding: 8px; background: var(--bg-secondary); border-radius: 4px; border-left: 3px solid var(--brand-accent);">
                            <strong style="color: var(--brand-accent);">KI-Bewertung:</strong> {evaluation_text}
                        </div>
            """
        
        html_content += f"""
                        <strong>Textvorschau:</strong><br>
                        {content_preview}
                    </div>
                </div>
            </div>
        </div>
        """
    
    html_content += "</div>"
    
    return html_content

def update_chunks_display(retrieved_chunks: Dict[str, Any]) -> tuple:
    """
    Update the chunks display with retrieved results.
    UPDATED: Uses CSS message classes for status updates.
    
    Args:
        retrieved_chunks: Retrieved chunks data
        
    Returns:
        Tuple of updated components
    """
    if not retrieved_chunks or not retrieved_chunks.get('chunks'):
        empty_message = "<div class='info-message'><p><em>Keine Texte verfügbar. Führen Sie zuerst eine Heuristik durch.</em></p></div>"
        return (
            empty_message,
            "**Noch keine Texte verfügbar**",
            gr.update(visible=False),  # select_all_btn
            gr.update(visible=False),  # deselect_all_btn
            gr.update(visible=False),  # transfer_btn
            [],  # available_chunks_state
            []   # selected_chunk_ids_state
        )
    
    chunks = retrieved_chunks.get('chunks', [])
    
    # Format chunks with checkboxes using CSS classes
    chunks_html = format_chunks_with_checkboxes(retrieved_chunks)
    
    # Initialize with all chunks selected
    all_chunk_ids = list(range(1, len(chunks) + 1))
    
    # Create summary with better formatting
    search_method = retrieved_chunks.get('metadata', {}).get('retrieval_method', 'standard')
    method_display = "LLM-Unterstützte Auswahl" if 'llm_assisted' in search_method else "Standard-Heuristik"
    
    summary_text = f"""**Verfügbare Texte**: {len(chunks)} ({method_display}) | **Ausgewählt**: {len(chunks)} (alle)

Wählen Sie die Texte aus, die Sie für die Analyse verwenden möchten."""
    
    return (
        chunks_html,
        summary_text,
        gr.update(visible=True),   # select_all_btn
        gr.update(visible=True),   # deselect_all_btn
        gr.update(visible=True),   # transfer_btn
        chunks,                    # available_chunks_state
        all_chunk_ids             # selected_chunk_ids_state
    )

def handle_select_all(available_chunks: List[Dict]) -> tuple:
    """
    Handle select all button click.
    UPDATED: Better status messaging with CSS classes.
    """
    if not available_chunks:
        return (
            "**Keine Texte verfügbar**",
            []
        )
    
    all_ids = list(range(1, len(available_chunks) + 1))
    summary = f"""**Verfügbare Texte**: {len(available_chunks)} | **Ausgewählt**: {len(available_chunks)} (alle)

✅ Alle Texte wurden ausgewählt."""
    
    return (summary, all_ids)

def handle_deselect_all(available_chunks: List[Dict]) -> tuple:
    """
    Handle deselect all button click.
    UPDATED: Better status messaging.
    """
    if not available_chunks:
        return (
            "**Keine Texte verfügbar**",
            []
        )
    
    summary = f"""**Verfügbare Texte**: {len(available_chunks)} | **Ausgewählt**: 0 (keine)

❌ Alle Texte wurden abgewählt."""
    
    return (summary, [])

def transfer_chunks_to_analysis(
    available_chunks: List[Dict], 
    selected_chunk_ids: List[int]
) -> tuple:
    """
    Transfer selected chunks to analysis section.
    UPDATED: Uses CSS message classes for better styling.
    
    Args:
        available_chunks: All available chunks
        selected_chunk_ids: IDs of selected chunks
        
    Returns:
        Tuple of transfer status and transferred chunks
    """
    if not available_chunks:
        error_message = """<div class="error-message">
        <h4>❌ Keine Texte verfügbar</h4>
        <p>Führen Sie zuerst eine Heuristik durch, um Texte zum Übertragen zu erhalten.</p>
        </div>"""
        return (
            gr.update(value=error_message, visible=True),
            []
        )
    
    if not selected_chunk_ids:
        warning_message = """<div class="warning-message">
        <h4>⚠️ Keine Texte ausgewählt</h4>
        <p>Wählen Sie mindestens einen Text aus, bevor Sie ihn zur Analyse übertragen.</p>
        </div>"""
        return (
            gr.update(value=warning_message, visible=True),
            []
        )
    
    # Filter chunks by selected IDs
    transferred_chunks = []
    for chunk_id in selected_chunk_ids:
        # Convert to 0-based index
        index = chunk_id - 1
        if 0 <= index < len(available_chunks):
            chunk = available_chunks[index].copy()
            chunk['transferred_id'] = chunk_id
            transferred_chunks.append(chunk)
    
    if not transferred_chunks:
        error_message = """<div class="error-message">
        <h4>❌ Ungültige Auswahl</h4>
        <p>Die ausgewählten Text-IDs sind nicht gültig. Bitte versuchen Sie es erneut.</p>
        </div>"""
        return (
            gr.update(value=error_message, visible=True),
            []
        )
    
    # UPDATED: Create success message with CSS styling
    success_message = f"""<div class="success-message">
    <h4>✅ Texte erfolgreich übertragen</h4>
    <p><strong>{len(transferred_chunks)} von {len(available_chunks)} Texten</strong> wurden zur Analyse übertragen.</p>
    
    <p><strong>Nächste Schritte:</strong></p>
    <ul>
        <li>Wechseln Sie zum <strong>"Analyse"-Tab</strong></li>
        <li>Die übertragenen Texte sind dort bereits vorausgewählt</li>
        <li>Formulieren Sie Ihre Forschungsfrage</li>
        <li>Starten Sie die Analyse</li>
    </ul>
    
    <p><strong>Übertragene Text-IDs:</strong> {', '.join(map(str, selected_chunk_ids))}</p>
    </div>"""
    
    return (
        gr.update(value=success_message, visible=True),
        transferred_chunks
    )