# src/ui/utils/checkbox_handler.py - SIMPLIFIED: Working checkbox state management
"""
SIMPLIFIED: Working checkbox state management that actually functions.
Removed unnecessary complexity and focuses on what works.
"""
import gradio as gr
import json
from typing import List, Dict, Any, Tuple

def create_checkbox_state_handler() -> Dict[str, Any]:
    """
    SIMPLIFIED: Create a working checkbox state handler.
    """
    # Simple hidden textbox to receive checkbox state updates
    checkbox_states_input = gr.Textbox(
        value="",
        visible=False,
        elem_id="checkbox_states_input",
        interactive=True
    )
    
    return {
        "checkbox_states_input": checkbox_states_input
    }

def handle_checkbox_state_update(
    checkbox_states_json: str, 
    available_chunks: List[Dict],
    current_selected: List[int]
) -> Tuple[List[int], str]:
    """
    SIMPLIFIED: Handle checkbox state updates from JavaScript.
    
    Args:
        checkbox_states_json: JSON string with selected chunk IDs
        available_chunks: List of available chunks
        current_selected: Currently selected chunk IDs
        
    Returns:
        Tuple of (new_selected_ids, summary_text)
    """
    total_chunks = len(available_chunks)
    
    if total_chunks == 0:
        return [], "**Keine Texte verfügbar**"
    
    # Parse the checkbox states
    try:
        if checkbox_states_json.strip():
            # JavaScript sends us the selected IDs directly
            selected_ids = json.loads(checkbox_states_json)
            if isinstance(selected_ids, list):
                # Validate IDs are within range
                valid_ids = [int(id) for id in selected_ids if isinstance(id, (int, str)) and str(id).isdigit()]
                valid_ids = [id for id in valid_ids if 1 <= id <= total_chunks]
                valid_ids = sorted(list(set(valid_ids)))  # Remove duplicates and sort
            else:
                valid_ids = current_selected
        else:
            # No update, keep current selection
            valid_ids = current_selected
            
    except (json.JSONDecodeError, ValueError, TypeError):
        # Error parsing, keep current selection
        valid_ids = current_selected
    
    # Create summary text
    if total_chunks == 0:
        summary_text = "**Keine Texte verfügbar**"
    else:
        percentage = (len(valid_ids) / total_chunks * 100) if total_chunks > 0 else 0
        summary_text = f"**Verfügbare Texte**: {total_chunks} | **Ausgewählt**: {len(valid_ids)}"
        
        if len(valid_ids) == total_chunks:
            summary_text += " (alle) ✅"
        elif len(valid_ids) == 0:
            summary_text += " (keine) ❌"
        else:
            summary_text += f" ({percentage:.0f}%) 📊"
    
    return valid_ids, summary_text

# SIMPLIFIED: Working JavaScript integration for the chunks display
WORKING_CHECKBOX_JAVASCRIPT = """
<script>
// SIMPLIFIED: Working checkbox management that actually syncs with Gradio
let selectedChunkIds = [];

function updateChunkSelection() {
    const checkboxes = document.querySelectorAll('input[name="chunk_selection"]');
    const checkedBoxes = document.querySelectorAll('input[name="chunk_selection"]:checked');
    selectedChunkIds = Array.from(checkedBoxes).map(cb => parseInt(cb.value));
    
    // Update visual summary
    updateSelectionSummary(checkboxes.length, selectedChunkIds.length);
    
    // CRITICAL: Update the hidden Gradio input to sync state
    const hiddenInput = document.getElementById('checkbox_states_input');
    if (hiddenInput) {
        hiddenInput.value = JSON.stringify(selectedChunkIds);
        
        // Trigger Gradio's change event
        const event = new Event('input', { bubbles: true });
        hiddenInput.dispatchEvent(event);
    }
    
    return selectedChunkIds;
}

function updateSelectionSummary(total, selected) {
    // Find summary elements (try multiple selectors)
    const summarySelectors = [
        '#chunks_selection_summary p',
        '[data-testid="markdown"] p',
        '.selection-summary'
    ];
    
    let summaryElement = null;
    for (const selector of summarySelectors) {
        summaryElement = document.querySelector(selector);
        if (summaryElement) break;
    }
    
    if (summaryElement) {
        let status = `**Verfügbare Texte**: ${total} | **Ausgewählt**: ${selected}`;
        if (selected === total) status += ' (alle) ✅';
        else if (selected === 0) status += ' (keine) ❌';
        else status += ` (${Math.round(selected/total*100)}%) 📊`;
        
        summaryElement.innerHTML = status;
    }
}

function selectAllChunks() {
    document.querySelectorAll('input[name="chunk_selection"]').forEach(cb => {
        cb.checked = true;
    });
    updateChunkSelection();
}

function deselectAllChunks() {
    document.querySelectorAll('input[name="chunk_selection"]').forEach(cb => {
        cb.checked = false;
    });
    updateChunkSelection();
}

// WORKING: Initialize and setup event listeners
function initializeCheckboxes() {
    // Set up change listeners for all checkboxes
    document.querySelectorAll('input[name="chunk_selection"]').forEach(checkbox => {
        checkbox.addEventListener('change', updateChunkSelection);
    });
    
    // Initial update
    updateChunkSelection();
}

// WORKING: Setup with multiple initialization strategies
document.addEventListener('DOMContentLoaded', initializeCheckboxes);

// Also try to initialize after a short delay (for dynamic content)
setTimeout(initializeCheckboxes, 500);
setTimeout(initializeCheckboxes, 1000);

// Watch for dynamic content changes
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.type === 'childList') {
            // Check if checkboxes were added
            const checkboxes = document.querySelectorAll('input[name="chunk_selection"]');
            if (checkboxes.length > 0) {
                initializeCheckboxes();
            }
        }
    });
});

// Start observing
observer.observe(document.body, {
    childList: true,
    subtree: true
});

// WORKING: Global function to get current selection (for transfer)
window.getCurrentSelection = function() {
    return selectedChunkIds;
};
</script>
"""