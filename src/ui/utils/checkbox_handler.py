# src/ui/utils/checkbox_handler.py - SIMPLIFIED: Just provides hidden input
"""
SIMPLIFIED: Minimal checkbox handler that just provides a hidden input for JavaScript communication.
All the complex state management is now handled explicitly with the confirmation button.
"""
import gradio as gr
from typing import Dict, Any

def create_checkbox_state_handler() -> Dict[str, Any]:
    """
    SIMPLIFIED: Just create a hidden input for JavaScript communication.
    No more complex state management - everything is explicit now.
    """
    # Simple hidden textbox for JavaScript to send selection data
    js_selection_input = gr.Textbox(
        value="",
        visible=False,
        elem_id="js_selection_input",
        interactive=True,
        label="JavaScript Selection Input"
    )
    
    return {
        "js_selection_input": js_selection_input
    }
