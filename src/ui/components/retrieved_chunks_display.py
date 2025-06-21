# src/ui/components/retrieved_chunks_display.py
"""
FIXED: Simplified chunk selection with explicit state confirmation.
- Added "Auswahl bestätigen" button to explicitly read checkbox state
- Simplified JavaScript to just manage visual state
- Clear separation between visual interaction and state management
"""
import gradio as gr
from typing import Dict, List, Any, Optional
import json

def create_fixed_retrieved_chunks_display() -> Dict[str, Any]:
    """Create WORKING chunks display with explicit state confirmation."""
    
    with gr.Group(elem_classes=["form-container"]):
        gr.HTML("<h3 style='margin-top: 0; color: var(--text-primary);'>📄 Gefundene Texte</h3>")
        
        # Main display area
        chunks_selection_html = gr.HTML(
            value="<div class='info-message'><p><em>Führen Sie zuerst eine Heuristik durch...</em></p></div>"
        )
        
        # Selection summary and controls
        with gr.Row():
            selection_summary = gr.Markdown("**Noch keine Texte verfügbar**")
        
        with gr.Row():
            select_all_btn = gr.Button("✅ Alle auswählen", size="sm", visible=False)
            deselect_all_btn = gr.Button("❌ Alle abwählen", size="sm", visible=False)
            
        # NEW: Explicit confirmation button
        with gr.Row():
            confirm_selection_btn = gr.Button(
                "🔍 Auswahl bestätigen", 
                variant="secondary", 
                size="sm",
                visible=False,
                info="Liest die aktuellen Checkbox-Einstellungen und aktualisiert die Auswahl"
            )
        
        # Transfer button
        transfer_to_analysis_btn = gr.Button(
            "🔄 Bestätigte Quellen zur Analyse übertragen",
            variant="primary",
            visible=False
        )
        
        transfer_status = gr.Markdown(value="", visible=False)
        
        # SIMPLIFIED: State management
        available_chunks_state = gr.State([])
        confirmed_selection_state = gr.State([])  # NEW: Explicitly confirmed selection
        transferred_chunks_state = gr.State([])
        
        # Hidden input for JavaScript communication
        js_selection_input = gr.Textbox(
            value="",
            visible=False,
            elem_id="js_selection_input",  # Important: JavaScript needs this ID
            interactive=True
        )
        
        # Simple JavaScript with button integration
        gr.HTML("""
<script>
// SIMPLIFIED: Visual checkbox management with button integration
function updateVisualSummary() {
    const checkboxes = document.querySelectorAll('input[name="chunk_selection"]');
    const checkedBoxes = document.querySelectorAll('input[name="chunk_selection"]:checked');
    
    const total = checkboxes.length;
    const selected = checkedBoxes.length;
    
    if (total > 0) {
        let status = `**Verfügbare Texte**: ${total} | **Visuell ausgewählt**: ${selected}`;
        if (selected === total) status += ' (alle)';
        else if (selected === 0) status += ' (keine)';
        else status += ` (${Math.round(selected/total*100)}%)`;
        status += ' - Klicken Sie "Auswahl bestätigen" um die Auswahl zu übernehmen';
        
        // Update the summary display
        const summaryElement = document.querySelector('#selection_summary p');
        if (summaryElement) {
            summaryElement.innerHTML = status;
        }
    }
}

function selectAllChunks() {
    document.querySelectorAll('input[name="chunk_selection"]').forEach(cb => cb.checked = true);
    updateVisualSummary();
}

function deselectAllChunks() {
    document.querySelectorAll('input[name="chunk_selection"]').forEach(cb => cb.checked = false);
    updateVisualSummary();
}

// Function to get current selection and update hidden input
function confirmCurrentSelection() {
    const checkedBoxes = document.querySelectorAll('input[name="chunk_selection"]:checked');
    const selectedIds = Array.from(checkedBoxes).map(cb => parseInt(cb.value));
    
    // Send to hidden Gradio input
    const hiddenInput = document.getElementById('js_selection_input');
    if (hiddenInput) {
        hiddenInput.value = JSON.stringify(selectedIds);
        
        // Trigger Gradio's change event
        const event = new Event('input', { bubbles: true });
        hiddenInput.dispatchEvent(event);
    }
    
    return selectedIds;
}

// Initialize checkboxes
function initializeChunks() {
    document.querySelectorAll('input[name="chunk_selection"]').forEach(checkbox => {
        checkbox.addEventListener('change', updateVisualSummary);
    });
    updateVisualSummary();
}

// FIXED: Auto-trigger JavaScript functions when Gradio buttons are clicked
function setupButtonHandlers() {
    // Find buttons by their text content and add click handlers
    const buttons = document.querySelectorAll('button');
    
    buttons.forEach(button => {
        const buttonText = button.textContent.trim();
        
        if (buttonText === '✅ Alle auswählen') {
            button.addEventListener('click', function(e) {
                setTimeout(selectAllChunks, 10); // Small delay to let Gradio process first
            });
        }
        else if (buttonText === '❌ Alle abwählen') {
            button.addEventListener('click', function(e) {
                setTimeout(deselectAllChunks, 10); // Small delay to let Gradio process first
            });
        }
        else if (buttonText === '🔍 Auswahl bestätigen') {
            button.addEventListener('click', function(e) {
                setTimeout(confirmCurrentSelection, 10); // Small delay to let Gradio process first
            });
        }
    });
}

// Setup with multiple strategies
document.addEventListener('DOMContentLoaded', function() {
    initializeChunks();
    setupButtonHandlers();
});

setTimeout(function() {
    initializeChunks();
    setupButtonHandlers();
}, 500);

setTimeout(function() {
    initializeChunks();
    setupButtonHandlers();
}, 2000);

// Watch for dynamic content
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.type === 'childList') {
            const checkboxes = document.querySelectorAll('input[name="chunk_selection"]');
            if (checkboxes.length > 0) {
                initializeChunks();
            }
            
            // Also check for new buttons
            const buttons = document.querySelectorAll('button');
            if (buttons.length > 0) {
                setupButtonHandlers();
            }
        }
    });
});

observer.observe(document.body, { childList: true, subtree: true });

// Global access for debugging
window.selectAllChunks = selectAllChunks;
window.deselectAllChunks = deselectAllChunks;
window.confirmCurrentSelection = confirmCurrentSelection;
</script>
        """, visible=False)
    
    return {
        "chunks_selection_html": chunks_selection_html,
        "selection_summary": selection_summary,
        "select_all_btn": select_all_btn,
        "deselect_all_btn": deselect_all_btn,
        "confirm_selection_btn": confirm_selection_btn,  # NEW
        "transfer_to_analysis_btn": transfer_to_analysis_btn,
        "transfer_status": transfer_status,
        "available_chunks_state": available_chunks_state,
        "confirmed_selection_state": confirmed_selection_state,  # NEW
        "transferred_chunks_state": transferred_chunks_state,
        "js_selection_input": js_selection_input  # NEW
    }

def format_chunks_with_checkboxes(retrieved_chunks: Dict[str, Any]) -> str:
    """SIMPLIFIED: Format chunks with simple checkboxes."""
    if not retrieved_chunks or not retrieved_chunks.get('chunks'):
        return "<div class='info-message'><p><em>Keine Texte verfügbar.</em></p></div>"
    
    chunks = retrieved_chunks.get('chunks', [])
    
    html_content = """<div style='max-height: 80vh; overflow-y: auto; padding: 10px;'>"""
    
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get('metadata', {})
        content = chunk.get('content', '')
        relevance_score = chunk.get('relevance_score', 0.0)
        
        # Get additional scores if available
        vector_score = chunk.get('vector_similarity_score', relevance_score)
        llm_score = chunk.get('llm_evaluation_score', None)
        
        # Metadata
        title = metadata.get('Artikeltitel', 'Kein Titel')
        date = metadata.get('Datum', 'Unbekannt')
        year = metadata.get('Jahrgang', 'Unbekannt')
        url = metadata.get('URL', '')
        
        html_content += f"""
        <div style="
            background: var(--bg-tertiary); 
            border: 1px solid var(--border-primary); 
            border-radius: 8px; 
            padding: 15px; 
            margin-bottom: 15px;
            border-left: 4px solid var(--brand-secondary);
        ">
            <div style="display: flex; align-items: flex-start; gap: 12px; margin-bottom: 12px;">
                <input 
                    type="checkbox" 
                    name="chunk_selection" 
                    value="{i}" 
                    checked 
                    onchange="updateVisualSummary()"
                    style="accent-color: var(--brand-primary); transform: scale(1.3); margin-top: 2px;"
                >
                <div style="flex: 1;">
                    <div style="color: var(--text-primary); font-weight: 600; font-size: 16px; margin-bottom: 6px;">
                        {i}. {title}
                    </div>
                    <div style="color: var(--text-secondary); font-size: 14px; margin-bottom: 10px;">
                        <strong>Datum:</strong> {date} | 
                        <strong>Jahr:</strong> {year} | 
                        <strong>Relevanz:</strong> {relevance_score:.3f}
        """
        
        # Add dual scores if available
        if llm_score is not None:
            html_content += f"""<br>
                        <span style="color: var(--brand-accent);">
                            <strong>Vector:</strong> {vector_score:.3f} | 
                            <strong>LLM:</strong> {llm_score:.3f}
                        </span>"""
        
        # Add URL if available
        if url and url != 'Keine URL':
            html_content += f"""<br>
                        <a href="{url}" target="_blank" style="color: var(--brand-primary); text-decoration: none;">🔗 Artikel öffnen</a>"""
        
        html_content += """
                    </div>
                </div>
            </div>
        """
        
        # Show evaluation text if available
        evaluation_text = metadata.get('evaluation_text', '')
        if evaluation_text:
            html_content += f"""
            <div style="
                background: var(--bg-primary); 
                border-left: 3px solid var(--brand-accent); 
                padding: 10px; 
                border-radius: 4px; 
                margin-bottom: 10px;
                color: var(--text-secondary);
            ">
                <strong style="color: var(--brand-accent);">🤖 KI-Bewertung:</strong> {evaluation_text}
            </div>
            """
        
        # Show full text content
        html_content += f"""
            <div style="
                background: var(--bg-primary); 
                border-left: 3px solid var(--brand-secondary); 
                padding: 12px; 
                border-radius: 4px;
                color: var(--text-secondary);
                line-height: 1.6;
                white-space: pre-wrap;
                max-height: 400px;
                overflow-y: auto;
            ">
                <strong style="color: var(--text-primary);">Volltext:</strong><br><br>
                {content}
            </div>
        </div>
        """
    
    html_content += "</div>"
    return html_content

def update_chunks_display(retrieved_chunks: Dict[str, Any]) -> tuple:
    """Update the chunks display with retrieved results."""
    if not retrieved_chunks or not retrieved_chunks.get('chunks'):
        return (
            "<div class='info-message'><p><em>Keine Texte verfügbar.</em></p></div>",
            "**Noch keine Texte verfügbar**",
            gr.update(visible=False),  # select_all_btn
            gr.update(visible=False),  # deselect_all_btn
            gr.update(visible=False),  # confirm_selection_btn
            gr.update(visible=False),  # transfer_btn
            [],  # available_chunks_state
            []   # confirmed_selection_state
        )
    
    chunks = retrieved_chunks.get('chunks', [])
    chunks_html = format_chunks_with_checkboxes(retrieved_chunks)
    
    search_method = retrieved_chunks.get('metadata', {}).get('retrieval_method', 'standard')
    method_display = "LLM-Unterstützte Auswahl" if 'llm_assisted' in search_method else "Standard-Heuristik"
    
    summary_text = f"**Verfügbare Texte**: {len(chunks)} ({method_display}) | **Visuell ausgewählt**: {len(chunks)} (alle) - Klicken Sie 'Auswahl bestätigen' um die Auswahl zu übernehmen"
    
    return (
        chunks_html,
        summary_text,
        gr.update(visible=True),   # select_all_btn
        gr.update(visible=True),   # deselect_all_btn
        gr.update(visible=True),   # confirm_selection_btn
        gr.update(visible=True),   # transfer_btn
        chunks,                    # available_chunks_state
        []                         # confirmed_selection_state (empty until confirmed)
    )

def handle_select_all(available_chunks: List[Dict]) -> str:
    """Handle select all - just updates visual state."""
    if not available_chunks:
        return "**Keine Texte verfügbar**"
    
    summary = f"**Verfügbare Texte**: {len(available_chunks)} | **Visuell ausgewählt**: {len(available_chunks)} (alle) - Klicken Sie 'Auswahl bestätigen' um die Auswahl zu übernehmen"
    return summary

def handle_deselect_all(available_chunks: List[Dict]) -> str:
    """Handle deselect all - just updates visual state."""
    if not available_chunks:
        return "**Keine Texte verfügbar**"
    
    summary = f"**Verfügbare Texte**: {len(available_chunks)} | **Visuell ausgewählt**: 0 (keine) - Klicken Sie 'Auswahl bestätigen' um die Auswahl zu übernehmen"
    return summary

def confirm_selection(js_selection_json: str, available_chunks: List[Dict]) -> tuple:
    """
    NEW: Confirm the current visual selection and update Gradio state.
    """
    if not available_chunks:
        return (
            [],  # confirmed_selection_state
            "**Keine Texte verfügbar**",  # selection_summary
            gr.update(visible=False)  # transfer_btn
        )
    
    try:
        # Parse the selection from JavaScript
        if js_selection_json.strip():
            selected_ids = json.loads(js_selection_json)
            if not isinstance(selected_ids, list):
                selected_ids = []
        else:
            # If no data from JavaScript, assume all are selected (default state)
            selected_ids = list(range(1, len(available_chunks) + 1))
        
        # Validate and filter the IDs
        valid_ids = []
        for chunk_id in selected_ids:
            if isinstance(chunk_id, (int, str)) and str(chunk_id).isdigit():
                chunk_id = int(chunk_id)
                if 1 <= chunk_id <= len(available_chunks):
                    valid_ids.append(chunk_id)
        
        valid_ids = sorted(list(set(valid_ids)))  # Remove duplicates and sort
        
        # Update summary
        if valid_ids:
            percentage = (len(valid_ids) / len(available_chunks) * 100)
            summary = f"**Verfügbare Texte**: {len(available_chunks)} | **Bestätigt ausgewählt**: {len(valid_ids)}"
            
            if len(valid_ids) == len(available_chunks):
                summary += " (alle) ✅"
            else:
                summary += f" ({percentage:.0f}%) ✅"
            
            summary += f" | **IDs**: {', '.join(map(str, valid_ids[:10]))}"
            if len(valid_ids) > 10:
                summary += "..."
            
            transfer_btn_state = gr.update(visible=True)
        else:
            summary = f"**Verfügbare Texte**: {len(available_chunks)} | **Bestätigt ausgewählt**: 0 (keine) ❌"
            transfer_btn_state = gr.update(visible=False)
        
        return (valid_ids, summary, transfer_btn_state)
        
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # Error parsing - default to all selected
        all_ids = list(range(1, len(available_chunks) + 1))
        summary = f"**Verfügbare Texte**: {len(available_chunks)} | **Bestätigt ausgewählt**: {len(all_ids)} (alle - Fallback wegen Lesefehler) ⚠️"
        return (all_ids, summary, gr.update(visible=True))

def transfer_chunks_to_analysis(
    available_chunks: List[Dict], 
    confirmed_selection: List[int]
) -> tuple:
    """
    SIMPLIFIED: Transfer confirmed chunks to analysis.
    """
    if not available_chunks:
        error_message = """<div class="error-message">
        <h4>❌ Keine Texte verfügbar</h4>
        <p>Führen Sie zuerst eine Heuristik durch.</p>
        </div>"""
        return (gr.update(value=error_message, visible=True), [])
    
    if not confirmed_selection:
        error_message = """<div class="error-message">
        <h4>❌ Keine Texte ausgewählt</h4>
        <p>Bestätigen Sie zuerst Ihre Auswahl mit "Auswahl bestätigen".</p>
        </div>"""
        return (gr.update(value=error_message, visible=True), [])
    
    # Filter chunks by confirmed selection
    transferred_chunks = []
    for chunk_id in confirmed_selection:
        index = chunk_id - 1  # Convert to 0-based index
        if 0 <= index < len(available_chunks):
            chunk = available_chunks[index].copy()
            chunk['transferred_id'] = chunk_id
            transferred_chunks.append(chunk)
    
    if not transferred_chunks:
        error_message = """<div class="error-message">
        <h4>❌ Übertragung fehlgeschlagen</h4>
        <p>Keine gültigen Texte in der bestätigten Auswahl.</p>
        </div>"""
        return (gr.update(value=error_message, visible=True), [])
    
    success_message = f"""<div class="success-message">
    <h4>✅ Texte erfolgreich übertragen</h4>
    <p><strong>{len(transferred_chunks)} von {len(available_chunks)} Texten</strong> wurden zur Analyse übertragen.</p>
    
    <p><strong>Übertragene Text-IDs:</strong> {', '.join(map(str, confirmed_selection))}</p>
    
    <p><em>Sie können jederzeit eine neue Auswahl treffen und erneut übertragen.</em></p>
    </div>"""
    
    return (
        gr.update(value=success_message, visible=True),
        transferred_chunks
    )