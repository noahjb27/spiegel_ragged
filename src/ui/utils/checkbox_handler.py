# src/ui/utils/checkbox_handler.py
"""
Handler for managing checkbox states in the retrieved chunks display.
UPDATED: Now uses the centralized CSS design system with dark theme support.
"""
import gradio as gr
import json
from typing import List, Dict, Any, Tuple

def create_checkbox_state_handler() -> Dict[str, Any]:
    """
    Create components for handling checkbox state updates.
    UPDATED: Uses main CSS classes for consistency.
    
    Returns:
        Dictionary containing state management components
    """
    
    # Hidden textbox to capture checkbox states via JavaScript
    checkbox_states_input = gr.Textbox(
        value="",
        visible=False,
        elem_id="checkbox_states_input"
    )
    
    # Button to trigger state update (hidden, triggered by JS)
    update_selection_btn = gr.Button(
        "Update Selection",
        visible=False,
        elem_id="update_selection_trigger"
    )
    
    return {
        "checkbox_states_input": checkbox_states_input,
        "update_selection_btn": update_selection_btn
    }

def parse_checkbox_states(checkbox_states_json: str, total_chunks: int) -> Tuple[List[int], str]:
    """
    Parse checkbox states from JSON and return selected IDs and summary.
    UPDATED: Better error handling and formatting.
    
    Args:
        checkbox_states_json: JSON string containing checkbox states
        total_chunks: Total number of available chunks
        
    Returns:
        Tuple of (selected_chunk_ids, summary_text)
    """
    try:
        if not checkbox_states_json.strip():
            # Default to all selected if no state provided
            selected_ids = list(range(1, total_chunks + 1))
        else:
            # Parse JSON to get selected IDs
            checkbox_data = json.loads(checkbox_states_json)
            if isinstance(checkbox_data, list):
                selected_ids = [int(x) for x in checkbox_data if str(x).isdigit()]
            else:
                selected_ids = list(range(1, total_chunks + 1))
        
        # Validate IDs are within range
        selected_ids = [id for id in selected_ids if 1 <= id <= total_chunks]
        
        # UPDATED: Better summary formatting
        summary_text = f"**Verfügbare Texte**: {total_chunks} | **Ausgewählt**: {len(selected_ids)}"
        if len(selected_ids) == total_chunks:
            summary_text += " (alle)"
        elif len(selected_ids) == 0:
            summary_text += " (keine)"
        else:
            percentage = (len(selected_ids) / total_chunks) * 100
            summary_text += f" ({percentage:.0f}%)"
        
        return selected_ids, summary_text
        
    except (json.JSONDecodeError, ValueError) as e:
        # Fallback to all selected on error
        selected_ids = list(range(1, total_chunks + 1))
        summary_text = f"**Verfügbare Texte**: {total_chunks} | **Ausgewählt**: {total_chunks} (alle - Parsing-Fehler)"
        return selected_ids, summary_text

def create_enhanced_chunks_html(chunks: List[Dict[str, Any]], selected_ids: List[int] = None) -> str:
    """
    Create enhanced HTML for chunks with dark theme CSS integration.
    UPDATED: Uses CSS classes and variables from main design system.
    
    Args:
        chunks: List of chunk data
        selected_ids: List of initially selected chunk IDs
        
    Returns:
        HTML string with CSS-consistent styling
    """
    if not chunks:
        return "<div class='info-message'><p><em>Keine Texte verfügbar.</em></p></div>"
    
    # Default to all selected if not specified
    if selected_ids is None:
        selected_ids = list(range(1, len(chunks) + 1))
    
    # UPDATED: Use CSS classes and variables instead of hardcoded styles
    html_content = f"""
    <div id="chunks-container" class="results-container" style="max-height: 600px; overflow-y: auto;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h3 style="margin: 0; color: var(--text-primary);">📄 Gefundene Texte ({len(chunks)})</h3>
            <div>
                <button onclick="selectAllChunks()" class="btn-secondary" style="margin-right: 10px;">
                    ✅ Alle auswählen
                </button>
                <button onclick="deselectAllChunks()" class="btn-secondary">
                    ❌ Alle abwählen
                </button>
            </div>
        </div>
        
        <div class="info-message" style="margin-bottom: 20px;">
            <p>Wählen Sie die Texte aus, die Sie für die Analyse verwenden möchten.</p>
        </div>
        
        <div id="selection-summary" class="analysis-info" style="margin-bottom: 15px;">
            <strong>Ausgewählt: {len(selected_ids)} von {len(chunks)} Texten</strong>
        </div>
    """
    
    for i, chunk in enumerate(chunks):
        chunk_id = i + 1
        metadata = chunk.get('metadata', {})
        title = metadata.get('Artikeltitel', 'Kein Titel')
        date = metadata.get('Datum', 'Unbekannt')
        year = metadata.get('Jahrgang', 'Unbekannt')
        relevance = chunk.get('relevance_score', 0.0)
        url = metadata.get('URL', '')
        
        # Get additional scores if available (LLM-assisted search)
        vector_score = chunk.get('vector_similarity_score', relevance)
        llm_score = chunk.get('llm_evaluation_score', None)
        evaluation_text = metadata.get('evaluation_text', '')
        
        # Get content
        content = chunk.get('content', '')
        content_preview = content[:400] + '...' if len(content) > 400 else content
        
        # Check if this chunk should be selected
        is_checked = "checked" if chunk_id in selected_ids else ""
        
        # Create URL link if available
        title_display = title
        if url and url != 'Keine URL':
            title_display = f'<a href="{url}" target="_blank" style="color: var(--brand-primary); text-decoration: none; font-weight: 600;">{title} 🔗</a>'
        else:
            title_display = f'<span style="color: var(--text-primary); font-weight: 600;">{title}</span>'
        
        # UPDATED: Use CSS classes for chunk containers
        html_content += f"""
        <div class="evaluation-card" style="margin-bottom: 15px;" id="chunk_container_{chunk_id}">
            <div style="margin-bottom: 12px;">
                <label style="display: flex; align-items: flex-start; cursor: pointer;">
                    <input type="checkbox" 
                           class="chunk-checkbox"
                           value="{chunk_id}" 
                           {is_checked}
                           onchange="updateChunkSelection()"
                           style="accent-color: var(--brand-primary); margin-right: 12px; margin-top: 2px; transform: scale(1.3);">
                    <div style="flex: 1;">
                        <div style="font-size: 16px; margin-bottom: 6px;">
                            <strong style="color: var(--text-primary);">{chunk_id}.</strong> {title_display}
                        </div>
                        <div style="font-size: 14px; color: var(--text-secondary); margin-bottom: 8px;">
                            <strong>Datum:</strong> {date} | 
                            <strong>Jahr:</strong> {year} | 
                            <strong>Relevanz:</strong> {relevance:.3f}
        """
        
        # Add dual scores if available (LLM-assisted search)
        if llm_score is not None:
            html_content += f"""<br>
                            <span style="color: var(--brand-accent);">
                                <strong>Vector:</strong> {vector_score:.3f} | 
                                <strong>LLM:</strong> {llm_score:.3f}
                            </span>"""
        
        html_content += """
                        </div>
                    </div>
                </label>
            </div>
        """
        
        # Add LLM evaluation if available
        if evaluation_text:
            html_content += f"""
            <div style="margin-left: 35px; margin-bottom: 10px;">
                <div style="background: var(--bg-primary); border-left: 3px solid var(--brand-accent); padding: 10px; border-radius: 4px;">
                    <strong style="color: var(--brand-accent);">🤖 KI-Bewertung:</strong>
                    <span style="color: var(--text-secondary);">{evaluation_text}</span>
                </div>
            </div>
            """
        
        # Content preview with CSS styling
        html_content += f"""
            <div style="margin-left: 35px;">
                <details style="margin-top: 8px;">
                    <summary style="color: var(--text-primary); font-weight: 500; cursor: pointer; padding: 5px 0;">
                        📄 Textvorschau anzeigen
                    </summary>
                    <div style="border-left: 3px solid var(--brand-secondary); padding-left: 12px; background: var(--bg-primary); padding: 12px; border-radius: 4px; margin-top: 8px;">
                        <div style="color: var(--text-secondary); line-height: 1.6; font-size: 14px; white-space: pre-wrap;">
{content_preview}
                        </div>
                    </div>
                </details>
            </div>
        </div>
        """
    
    html_content += """
        </div>
        
        <script>
        function updateChunkSelection() {
            const checkboxes = document.querySelectorAll('.chunk-checkbox');
            const checkedBoxes = document.querySelectorAll('.chunk-checkbox:checked');
            const selectedIds = Array.from(checkedBoxes).map(cb => parseInt(cb.value));
            
            // Update visual summary with CSS styling
            const summaryDiv = document.getElementById('selection-summary');
            if (summaryDiv) {
                const total = checkboxes.length;
                const selected = selectedIds.length;
                const percentage = total > 0 ? Math.round((selected / total) * 100) : 0;
                
                let statusText = `<strong>Ausgewählt: ${selected} von ${total} Texten`;
                if (selected === total) {
                    statusText += ' (alle)</strong>';
                } else if (selected === 0) {
                    statusText += ' (keine)</strong>';
                } else {
                    statusText += ` (${percentage}%)</strong>`;
                }
                
                summaryDiv.innerHTML = statusText;
            }
            
            // Update container styling based on selection
            checkboxes.forEach(checkbox => {
                const container = document.getElementById(`chunk_container_${checkbox.value}`);
                if (container) {
                    if (checkbox.checked) {
                        container.style.borderLeftColor = 'var(--brand-primary)';
                        container.style.backgroundColor = 'rgba(215, 84, 37, 0.05)';
                    } else {
                        container.style.borderLeftColor = 'var(--border-primary)';
                        container.style.backgroundColor = 'var(--bg-tertiary)';
                    }
                }
            });
            
            // Update hidden input for Gradio
            const hiddenInput = document.getElementById('checkbox_states_input');
            if (hiddenInput) {
                hiddenInput.value = JSON.stringify(selectedIds);
                
                // Trigger Gradio update
                const event = new Event('input', { bubbles: true });
                hiddenInput.dispatchEvent(event);
            }
        }
        
        function selectAllChunks() {
            const checkboxes = document.querySelectorAll('.chunk-checkbox');
            checkboxes.forEach(cb => cb.checked = true);
            updateChunkSelection();
        }
        
        function deselectAllChunks() {
            const checkboxes = document.querySelectorAll('.chunk-checkbox');
            checkboxes.forEach(cb => cb.checked = false);
            updateChunkSelection();
        }
        
        // Enhanced styling for dark theme
        function enhanceCheckboxStyling() {
            const style = document.createElement('style');
            style.textContent = `
                .chunk-checkbox {
                    accent-color: var(--brand-primary) !important;
                    width: 18px !important;
                    height: 18px !important;
                    cursor: pointer !important;
                }
                
                .chunk-checkbox:hover {
                    transform: scale(1.4) !important;
                    transition: transform 0.2s ease !important;
                }
                
                details summary {
                    transition: color 0.2s ease !important;
                }
                
                details summary:hover {
                    color: var(--brand-primary) !important;
                }
                
                details[open] summary {
                    color: var(--brand-primary) !important;
                }
                
                /* Button styling consistency */
                .btn-secondary {
                    background: linear-gradient(135deg, var(--brand-secondary) 0%, var(--brand-secondary-hover) 100%) !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 6px !important;
                    padding: 8px 12px !important;
                    font-size: 13px !important;
                    cursor: pointer !important;
                    transition: all 0.2s ease !important;
                }
                
                .btn-secondary:hover {
                    background: linear-gradient(135deg, var(--brand-secondary-hover) 0%, #756c63 100%) !important;
                    transform: translateY(-1px) !important;
                }
            `;
            document.head.appendChild(style);
        }
        
        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {
            enhanceCheckboxStyling();
            updateChunkSelection();
        });
        
        // Also trigger on any dynamic content changes
        setTimeout(() => {
            enhanceCheckboxStyling();
            updateChunkSelection();
        }, 100);
        </script>
    """
    
    return html_content

def handle_checkbox_state_update(
    checkbox_states_json: str, 
    available_chunks: List[Dict],
    current_selected: List[int]
) -> Tuple[List[int], str]:
    """
    Handle checkbox state updates from JavaScript.
    UPDATED: Improved error handling and status messages.
    
    Args:
        checkbox_states_json: JSON string with checkbox states
        available_chunks: List of available chunks
        current_selected: Currently selected chunk IDs
        
    Returns:
        Tuple of (new_selected_ids, summary_text)
    """
    total_chunks = len(available_chunks)
    
    if not checkbox_states_json.strip():
        # No update, return current state
        percentage = (len(current_selected) / total_chunks * 100) if total_chunks > 0 else 0
        summary = f"**Verfügbare Texte**: {total_chunks} | **Ausgewählt**: {len(current_selected)} ({percentage:.0f}%)"
        return current_selected, summary
    
    try:
        # Parse the JSON from JavaScript
        selected_ids = json.loads(checkbox_states_json)
        if not isinstance(selected_ids, list):
            selected_ids = current_selected
        
        # Validate and filter IDs
        valid_ids = [int(id) for id in selected_ids if isinstance(id, (int, str)) and str(id).isdigit()]
        valid_ids = [id for id in valid_ids if 1 <= id <= total_chunks]
        
        # Remove duplicates and sort
        valid_ids = sorted(list(set(valid_ids)))
        
        # UPDATED: Enhanced summary with better formatting
        percentage = (len(valid_ids) / total_chunks * 100) if total_chunks > 0 else 0
        summary_text = f"**Verfügbare Texte**: {total_chunks} | **Ausgewählt**: {len(valid_ids)}"
        
        if len(valid_ids) == total_chunks:
            summary_text += " (alle) ✅"
        elif len(valid_ids) == 0:
            summary_text += " (keine) ❌"
        else:
            summary_text += f" ({percentage:.0f}%) 📊"
        
        return valid_ids, summary_text
        
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # Return current state on error with warning
        error_summary = f"**Verfügbare Texte**: {total_chunks} | **Ausgewählt**: {len(current_selected)} ⚠️ (Update-Fehler)"
        return current_selected, error_summary

def create_selection_status_html(selected_count: int, total_count: int, has_dual_scores: bool = False) -> str:
    """
    Create HTML for selection status display.
    UPDATED: Uses CSS classes for consistent styling.
    
    Args:
        selected_count: Number of selected chunks
        total_count: Total number of chunks
        has_dual_scores: Whether chunks have dual scores (LLM-assisted)
        
    Returns:
        HTML string with selection status
    """
    if total_count == 0:
        return "<div class='warning-message'><p>Keine Texte verfügbar</p></div>"
    
    percentage = (selected_count / total_count * 100) if total_count > 0 else 0
    search_type = "LLM-Unterstützte Auswahl" if has_dual_scores else "Standard-Heuristik"
    
    if selected_count == 0:
        status_class = "warning-message"
        icon = "⚠️"
        message = "Keine Texte ausgewählt"
    elif selected_count == total_count:
        status_class = "success-message" 
        icon = "✅"
        message = "Alle Texte ausgewählt"
    else:
        status_class = "info-message"
        icon = "📊"
        message = f"{percentage:.0f}% der Texte ausgewählt"
    
    return f"""
    <div class="{status_class}">
        <h4>{icon} Auswahl-Status</h4>
        <p><strong>{selected_count} von {total_count} Texten</strong> - {message}</p>
        <p><em>Suchmethode: {search_type}</em></p>
    </div>
    """