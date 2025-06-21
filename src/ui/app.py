# src/ui/app.py
from typing import Any, Dict, List, Tuple
import gradio as gr
import logging
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.engine import SpiegelRAG
from src.ui.handlers.search_handlers import set_rag_engine, update_chunks_display_handler
from src.ui.components.search_panel import create_search_panel
from src.ui.components.question_panel import create_question_panel
from src.ui.components.results_panel import create_results_panel
from src.ui.components.info_panel import create_info_panel
from src.ui.handlers.search_handlers import (
    perform_retrieval_and_update_ui,
    perform_analysis_and_update_ui
)
from src.ui.handlers.agent_handlers import (
    set_rag_engine as set_llm_assisted_rag_engine,
    perform_llm_assisted_search_threaded,
    cancel_llm_assisted_search,
    create_llm_assisted_download_comprehensive
)
from src.ui.handlers.keyword_handlers import (
    set_embedding_service,
    find_similar_words,
    expand_boolean_expression
)
from src.ui.handlers.download_handlers import (
    create_download_json, 
    create_download_csv,
    format_download_summary
)

from src.ui.components.retrieved_chunks_display import (
    create_fixed_retrieved_chunks_display,
    update_chunks_display,
    handle_select_all,
    handle_deselect_all,
    confirm_selection,
    transfer_chunks_to_analysis
)

from src.ui.utils.ui_helpers import toggle_api_key_visibility
from src.ui.utils.checkbox_handler import (create_checkbox_state_handler)


from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def perform_analysis_and_update_ui_with_transferred_chunks(
    user_prompt: str,
    transferred_chunks: list,
    model_selection: str,
    system_prompt_template: str,
    system_prompt_text: str,
    temperature: float
) -> tuple:
    """
    Perform analysis using transferred chunks.
    
    Args:
        user_prompt: User's research question
        transferred_chunks: Chunks transferred from heuristic phase
        model_selection: Selected LLM model
        system_prompt_template: Template name
        system_prompt_text: Actual system prompt text
        temperature: Generation temperature
        
    Returns:
        Tuple of analysis results for UI update
    """
    if not transferred_chunks:
        return (
            "❌ Keine Quellen für die Analyse verfügbar. Bitte übertragen Sie zuerst Quellen aus der Heuristik.",
            "**Fehler**: Keine übertragenen Quellen",
            gr.update(open=True),   # Keep analysis accordion open
            gr.update(open=False)   # Keep results closed
        )
    
    # Convert transferred chunks to the format expected by existing analysis function
    retrieved_chunks_format = {
        'chunks': transferred_chunks,
        'metadata': {
            'total_chunks': len(transferred_chunks),
            'source': 'transferred_from_heuristic'
        }
    }
    
    # Use existing analysis function with adapted format
    return perform_analysis_and_update_ui(
        user_prompt=user_prompt,
        retrieved_chunks=retrieved_chunks_format,
        model_selection=model_selection,
        system_prompt_template=system_prompt_template,
        system_prompt_text=system_prompt_text,
        temperature=temperature,
        chunk_selection_mode="all",  # Use all transferred chunks
        selected_chunks_state=None   # Not needed since we use all transferred
    )

def create_app():
    """Create the updated Gradio application."""
    
    # Initialize RAG engine
    logger.info("Initializing RAG engine...")
    try:
        rag_engine = SpiegelRAG()
        logger.info("RAG engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        raise
    
    # Set the engine reference for handlers
    set_rag_engine(rag_engine)
    set_llm_assisted_rag_engine(rag_engine)
    
    # Set embedding service for keyword handlers
    if rag_engine.embedding_service:
        set_embedding_service(rag_engine.embedding_service)
        logger.info("Embedding service connected to keyword handlers")
    else:
        logger.warning("Embedding service not available")
    

    updated_css = """
/* Color System with Custom Palette */
:root {
    /* Custom Brand Colors */
    --brand-primary: #d75425;      /* Orange - main actions */
    --brand-primary-hover: #c04a20;
    --brand-primary-light: #fef7f0;
    
    --brand-secondary: #968d84;    /* Warm gray - secondary actions */
    --brand-secondary-hover: #857c73;
    --brand-secondary-light: #f4f1ee;
    
    --brand-accent: #b2b069;       /* Olive - success/accent */
    --brand-accent-hover: #a0a05c;
    --brand-accent-light: #f9f8f4;
    
    /* Dark Background System */
    --bg-primary: #1a1a1a;
    --bg-secondary: #2d2d2d;
    --bg-tertiary: #3a3a3a;
    --bg-elevated: #404040;
    
    /* Text Colors for Dark Background */
    --text-primary: #ffffff;       /* Main text */
    --text-secondary: #e5e5e5;     /* Secondary text */
    --text-muted: #b0b0b0;         /* Muted text */
    --text-subtle: #8a8a8a;        /* Very subtle text */
    
    /* Border Colors */
    --border-primary: #4a4a4a;
    --border-secondary: #5a5a5a;
    --border-accent: var(--brand-primary);
    
    /* State Colors */
    --success: #22c55e;
    --success-bg: #14532d;
    --warning: #f59e0b;
    --warning-bg: #451a03;
    --error: #ef4444;
    --error-bg: #7f1d1d;
    --info: #3b82f6;
    --info-bg: #1e3a8a;
}

/* Base Styles for Dark Theme */
.gradio-container {
    max-width: 1400px !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Ensure all text is visible on dark background */
* {
    color: var(--text-primary) !important;
}

/* Override Gradio's default text colors */
.gr-markdown,
.gr-markdown *,
.gr-textbox,
.gr-textbox *,
.gr-dropdown,
.gr-dropdown *,
.gr-slider,
.gr-slider *,
.gr-checkbox,
.gr-checkbox *,
.gr-radio,
.gr-radio * {
    color: var(--text-primary) !important;
}

/* =============================================================================
HEADERS & TYPOGRAPHY
============================================================================= */

/* Main Headers */
h1, .gr-markdown h1 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 32px !important;
    line-height: 1.2 !important;
    margin-bottom: 16px !important;
    border-bottom: 3px solid var(--brand-primary) !important;
    padding-bottom: 12px !important;
}

/* Section Headers */
h2, .gr-markdown h2 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 24px !important;
    line-height: 1.3 !important;
    margin: 24px 0 16px 0 !important;
    position: relative !important;
    padding-left: 20px !important;
}

h2::before {
    content: '' !important;
    position: absolute !important;
    left: 0 !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: 4px !important;
    height: 20px !important;
    background: var(--brand-primary) !important;
    border-radius: 2px !important;
}

h3, .gr-markdown h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 20px !important;
    margin: 20px 0 12px 0 !important;
}

h4, .gr-markdown h4 {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 18px !important;
    margin: 16px 0 8px 0 !important;
}

p, .gr-markdown p {
    color: var(--text-secondary) !important;
    line-height: 1.6 !important;
    margin-bottom: 12px !important;
}

strong, .gr-markdown strong {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* =============================================================================
BUTTONS WITH CUSTOM COLORS
============================================================================= */

/* Primary Buttons - Orange Theme */
button[variant="primary"],
.btn-primary {
    background: linear-gradient(135deg, var(--brand-primary) 0%, var(--brand-primary-hover) 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 24px !important;
    border: none !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(215, 84, 37, 0.3) !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
}

button[variant="primary"]:hover,
.btn-primary:hover {
    background: linear-gradient(135deg, var(--brand-primary-hover) 0%, #a63f1b 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(215, 84, 37, 0.4) !important;
}

/* Secondary Buttons - Warm Gray Theme */
.btn-secondary {
    background: linear-gradient(135deg, var(--brand-secondary) 0%, var(--brand-secondary-hover) 100%) !important;
    color: white !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    border: none !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 6px rgba(150, 141, 132, 0.3) !important;
}

.btn-secondary:hover {
    background: linear-gradient(135deg, var(--brand-secondary-hover) 0%, #756c63 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 10px rgba(150, 141, 132, 0.4) !important;
}

/* Success/Download Buttons - Olive Theme */
.download-button,
.btn-success {
    background: linear-gradient(135deg, var(--brand-accent) 0%, var(--brand-accent-hover) 100%) !important;
    color: white !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 18px !important;
    border: none !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 6px rgba(178, 176, 105, 0.3) !important;
    text-shadow: 0 1px 1px rgba(0, 0, 0, 0.2) !important;
}

.download-button:hover,
.btn-success:hover {
    background: linear-gradient(135deg, var(--brand-accent-hover) 0%, #8f8f52 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 10px rgba(178, 176, 105, 0.4) !important;
}

/* Cancel/Danger Buttons */
.cancel-button,
.btn-danger,
button[variant="stop"] {
    background: linear-gradient(135deg, var(--error) 0%, #dc2626 100%) !important;
    color: white !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 18px !important;
    border: none !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 6px rgba(239, 68, 68, 0.3) !important;
}

.cancel-button:hover,
.btn-danger:hover,
button[variant="stop"]:hover {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 10px rgba(239, 68, 68, 0.4) !important;
}

/* Disabled State */
button:disabled,
.btn-disabled {
    background: var(--bg-tertiary) !important;
    color: var(--text-subtle) !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
    transform: none !important;
}

/* =============================================================================
LAYOUT CONTAINERS
============================================================================= */

.search-mode-container {
    background: var(--bg-secondary) !important;
    border: 2px solid var(--brand-primary) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

.chunk-selection-container,
.form-container {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-primary) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    margin: 12px 0 !important;
}

.chunk-selection-container h4,
.form-container h4 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    margin-bottom: 12px !important;
    border-bottom: 1px solid var(--border-primary) !important;
    padding-bottom: 8px !important;
}

.results-container {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-primary) !important;
    border-radius: 8px !important;
    padding: 24px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
}

/* =============================================================================
SPECIALIZED CONTAINERS (Enhanced from component CSS)
============================================================================= */

/* LLM-Assisted Results Container */
.llm-assisted-results {
    padding: 20px !important;
    border-radius: 8px !important;
    border: 1px solid var(--brand-accent) !important;
    margin-top: 20px !important;
    background-color: var(--bg-secondary) !important;
}

/* Evaluation Cards */
.evaluation-card {
    border-left: 4px solid var(--brand-primary) !important;
    padding: 15px !important;
    margin-bottom: 15px !important;
    background-color: var(--bg-tertiary) !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
}

.evaluation-card h4 {
    color: var(--text-primary) !important;
    margin-bottom: 10px !important;
    font-weight: bold !important;
}

.evaluation-card p,
.evaluation-card div {
    color: var(--text-secondary) !important;
    line-height: 1.6 !important;
    margin: 8px 0 !important;
}

.evaluation-card strong {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* Relevance Level Variants */
.evaluation-card.high-relevance {
    background-color: var(--success-bg) !important;
    border-left-color: var(--success) !important;
}

.evaluation-card.medium-relevance {
    background-color: var(--warning-bg) !important;
    border-left-color: var(--warning) !important;
}

.evaluation-card.low-relevance {
    background-color: var(--bg-tertiary) !important;
    border-left-color: var(--brand-secondary) !important;
}

/* Metadata Section */
.metadata-section {
    background-color: var(--bg-tertiary) !important;
    padding: 15px !important;
    border-radius: 8px !important;
    border: 1px solid var(--border-primary) !important;
    margin-top: 20px !important;
}

.metadata-section h3 {
    color: var(--text-primary) !important;
    margin-top: 0 !important;
}

/* Analysis Info Box */
.analysis-info {
    background: linear-gradient(135deg, var(--brand-accent-light) 0%, var(--bg-tertiary) 100%) !important;
    padding: 15px !important;
    border-radius: 8px !important;
    border-left: 4px solid var(--brand-accent) !important;
    margin-bottom: 20px !important;
}

.analysis-info h4 {
    color: var(--text-primary) !important;
    margin-bottom: 10px !important;
    font-weight: bold !important;
}

/* =============================================================================
ACCORDIONS & NAVIGATION
============================================================================= */

/* Accordion Headers */
.label-wrap {
    background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--bg-elevated) 100%) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 16px 20px !important;
    border: 2px solid var(--border-primary) !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.label-wrap:hover {
    background: linear-gradient(135deg, var(--bg-elevated) 0%, #4a4a4a 100%) !important;
    border-color: var(--brand-primary) !important;
    color: var(--text-primary) !important;
}

.label-wrap.open,
.label-wrap[aria-expanded="true"] {
    background: linear-gradient(135deg, var(--brand-primary) 0%, var(--brand-primary-hover) 100%) !important;
    color: white !important;
    border-color: var(--brand-primary) !important;
    border-radius: 8px 8px 0 0 !important;
}

/* Tab Navigation */
.tab-nav {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-primary) !important;
    border-radius: 8px 8px 0 0 !important;
}

.tab-nav button {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 14px 20px !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.tab-nav button:hover {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
}

.tab-nav button.selected,
.tab-nav button[aria-selected="true"] {
    background: var(--brand-primary) !important;
    color: white !important;
    font-weight: 600 !important;
}

/* =============================================================================
STATUS MESSAGES & FEEDBACK
============================================================================= */

/* Success Messages */
.success-message,
.interval-info {
    background: var(--success-bg) !important;
    border: 1px solid var(--success) !important;
    border-left: 4px solid var(--success) !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    margin: 12px 0 !important;
    color: #bbf7d0 !important;
}

/* Error Messages */
.error-message {
    background: var(--error-bg) !important;
    border: 1px solid var(--error) !important;
    border-left: 4px solid var(--error) !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    margin: 12px 0 !important;
    color: #fecaca !important;
}

/* Warning Messages */
.warning-message {
    background: var(--warning-bg) !important;
    border: 1px solid var(--warning) !important;
    border-left: 4px solid var(--warning) !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    margin: 12px 0 !important;
    color: #fed7aa !important;
}

/* Info Messages */
.info-message {
    background: var(--info-bg) !important;
    border: 1px solid var(--info) !important;
    border-left: 4px solid var(--info) !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    margin: 12px 0 !important;
    color: #bfdbfe !important;
}

/* Progress Indicators */
.llm-assisted-progress,
.progress-container {
    background: rgba(178, 176, 105, 0.15) !important;
    border: 1px solid var(--brand-accent) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    margin: 12px 0 !important;
    position: relative !important;
}

.llm-assisted-progress h4,
.progress-container h4 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
}

/* Filter Progress Visualization */
.filter-stage {
    margin-bottom: 20px !important;
    background-color: var(--bg-tertiary) !important;
    padding: 15px !important;
    border-radius: 8px !important;
    border: 1px solid var(--border-primary) !important;
}

.filter-stage-title {
    font-weight: bold !important;
    margin-bottom: 8px !important;
    color: var(--text-primary) !important;
    font-size: 16px !important;
}

.filter-progress {
    height: 30px !important;
    background-color: var(--bg-primary) !important;
    border-radius: 15px !important;
    overflow: hidden !important;
    margin-bottom: 5px !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
}

.filter-bar {
    height: 100% !important;
    background: linear-gradient(90deg, var(--brand-primary) 0%, var(--brand-accent) 100%) !important;
    text-align: center !important;
    color: white !important;
    line-height: 30px !important;
    font-weight: bold !important;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* =============================================================================
FORM ELEMENTS
============================================================================= */

/* Input Fields */
input[type="text"],
input[type="number"],
textarea,
select,
.gr-textbox input,
.gr-textbox textarea {
    background: var(--bg-tertiary) !important;
    border: 2px solid var(--border-primary) !important;
    border-radius: 6px !important;
    padding: 10px 12px !important;
    font-size: 14px !important;
    color: var(--text-primary) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

input:focus,
textarea:focus,
select:focus,
.gr-textbox input:focus,
.gr-textbox textarea:focus {
    outline: none !important;
    border-color: var(--brand-primary) !important;
    box-shadow: 0 0 0 3px rgba(215, 84, 37, 0.2) !important;
}

/* Labels */
label,
.gr-label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    margin-bottom: 6px !important;
}

/* Dropdown specific */
.gr-dropdown .choices__inner {
    background: var(--bg-tertiary) !important;
    border: 2px solid var(--border-primary) !important;
    color: var(--text-primary) !important;
}

/* =============================================================================
LLM ASSISTANT PROMPT CONTAINER (from component CSS)
============================================================================= */

.llm-assisted-prompt-container {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--brand-accent) !important;
    border-left: 4px solid var(--brand-accent) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    margin: 12px 0 !important;
}

.llm-assisted-prompt-container h4 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    margin-bottom: 12px !important;
}

/* Explanation Box */
.explanation-box {
    background-color: var(--bg-tertiary) !important;
    padding: 15px !important;
    margin-bottom: 20px !important;
    border-radius: 8px !important;
    border-left: 4px solid var(--brand-accent) !important;
}

.explanation-box h4 {
    color: var(--text-primary) !important;
    margin-bottom: 10px !important;
    font-weight: bold !important;
}

.explanation-box ul, .explanation-box li {
    color: var(--text-secondary) !important;
    line-height: 1.6 !important;
}

.explanation-box strong {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.explanation-box em {
    color: var(--brand-accent) !important;
    font-style: italic !important;
}

/* =============================================================================
CODE & CONTENT
============================================================================= */

/* Code Blocks */
pre,
code {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-primary) !important;
    border-radius: 4px !important;
    padding: 8px 12px !important;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
    font-size: 13px !important;
    color: var(--text-secondary) !important;
}

/* Tables */
table {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 16px 0 !important;
}

th,
td {
    border: 1px solid var(--border-primary) !important;
    padding: 8px 12px !important;
    text-align: left !important;
}

th {
    background: var(--bg-tertiary) !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

td {
    color: var(--text-secondary) !important;
}

/* Links */
a {
    color: var(--brand-primary) !important;
    text-decoration: none !important;
    font-weight: 500 !important;
    transition: color 0.2s ease !important;
}

a:hover {
    color: var(--brand-primary-hover) !important;
    text-decoration: underline !important;
}

/* Lists */
ul,
ol {
    color: var(--text-secondary) !important;
    line-height: 1.6 !important;
    margin-bottom: 16px !important;
    padding-left: 24px !important;
}

li {
    margin-bottom: 4px !important;
    color: var(--text-secondary) !important;
}

/* Quote styling */
blockquote {
    border-left: 4px solid var(--brand-accent) !important;
    padding-left: 1em !important;
    margin-left: 0 !important;
    font-style: italic !important;
    background-color: var(--bg-tertiary) !important;
    padding: 10px 15px !important;
    border-radius: 0 5px 5px 0 !important;
    color: var(--text-secondary) !important;
}

/* Source citation styling */
.citation {
    background-color: var(--bg-tertiary) !important;
    padding: 5px 10px !important;
    border-radius: 3px !important;
    border-left: 3px solid var(--brand-secondary) !important;
    margin: 5px 0 !important;
    font-size: 0.9em !important;
    color: var(--text-secondary) !important;
}

/* =============================================================================
RESPONSIVE DESIGN
============================================================================= */

@media (max-width: 768px) {
    .gradio-container {
        padding: 8px !important;
    }
    
    h1 {
        font-size: 24px !important;
    }
    
    h2 {
        font-size: 20px !important;
    }
    
    .btn-primary,
    .btn-secondary {
        padding: 10px 16px !important;
        font-size: 13px !important;
    }
    
    .search-mode-container,
    .chunk-selection-container {
        padding: 12px !important;
        margin-bottom: 16px !important;
    }
}

/* =============================================================================
UTILITY CLASSES
============================================================================= */

.text-center { text-align: center !important; }
.text-right { text-align: right !important; }
.mb-2 { margin-bottom: 16px !important; }
.mt-2 { margin-top: 16px !important; }
.p-2 { padding: 16px !important; }
.font-bold { font-weight: 700 !important; }
.text-primary { color: var(--text-primary) !important; }
.text-secondary { color: var(--text-secondary) !important; }
    """

    # Create the Gradio interface
    with gr.Blocks(
        title="SPIEGEL RAG System",  # UPDATED: Simplified title
        theme=gr.themes.Soft(),
        css=updated_css
    ) as app:
        
        # UPDATED: Main header with new terminology
        gr.Markdown("""
        # SPIEGEL RAG System (1948-1979)
        
        Ein Retrieval Augmented Generation (RAG) System zur Analyse und Durchsuchung des Spiegel-Archivs.
        
        **Systemstatus:** ✅ Verbunden mit ChromaDB und Ollama Embedding Service
        
        ## Funktionen
        
        - **Heuristik**: Getrennte Retrieval- und Analyse-Phasen für bessere Kontrolle
        - **LLM-Unterstützte Auswahl**: KI-gestützte Quellenbewertung mit anpassbaren Prompts
        - **Zeit-Interval-Suche**: Gleichmäßige zeitliche Verteilung der Quellen
        - **Quellenauswahl**: Interaktive Auswahl der zu analysierenden Texte
        """)
        
        # UPDATED: Tab structure with new names
        with gr.Tab("Heuristik"):  # UPDATED: Changed from "Quellen abrufen"
            with gr.Accordion("Suchmethode wählen", open=True, elem_classes=["search-mode-container"]) as search_method_accordion:
                # Create updated search panel components
                search_components = create_search_panel(
                    retrieve_callback=None,  # Will be set below
                    llm_assisted_search_callback=None,  # Will be set below
                    preview_callback=expand_boolean_expression,
                    toggle_api_key_callback=toggle_api_key_visibility
                )
            
            with gr.Accordion("Gefundene Texte", open=False) as retrieved_texts_accordion:
                gr.Markdown("""
                ### Gefundene Texte mit Auswahlmöglichkeit
                
                Hier werden die durch die Heuristik gefundenen Texte angezeigt. 
                Sie können einzelne Texte an- oder abwählen und dann zur Analyse übertragen.
                """)
                
                # NEW: Interactive chunks display component
                chunks_display_components = create_fixed_retrieved_chunks_display()
                
                # Add the checkbox state handler
                from src.ui.utils.checkbox_handler import create_checkbox_state_handler
                checkbox_handler = create_checkbox_state_handler()
                
                # Download functionality (existing, but moved below transfer section)
                gr.Markdown("---")
                gr.Markdown("### Downloads")
                
                with gr.Row():
                    download_json_btn = gr.Button("📥 Als JSON herunterladen", elem_classes=["download-button"])
                    download_csv_btn = gr.Button("📊 Als CSV herunterladen", elem_classes=["download-button"])
                    download_comprehensive_btn = gr.Button(
                        "📋 Umfassender LLM-Download", 
                        elem_classes=["download-button"],
                        visible=False
                    )
                
                # Download status and files (existing code)
                download_status = gr.Markdown("", visible=False)
                download_json_file = gr.File(label="JSON Download", visible=False)
                download_csv_file = gr.File(label="CSV Download", visible=False)
                download_comprehensive_file = gr.File(label="Comprehensive LLM Download", visible=False)
        
        with gr.Tab("Analyse"):
            with gr.Accordion("Analyse konfigurieren", open=True) as analysis_accordion:
                # UPDATED: Create question panel with transferred chunks support
                question_components = create_question_panel()
            
            with gr.Accordion("Ergebnisse", open=False) as results_accordion:
                # Create results panel components (existing code)
                results_components = create_results_panel()
                
                # Download analysis results (existing code)
                with gr.Row():
                    download_analysis_btn = gr.Button("📄 Analyse als TXT herunterladen", elem_classes=["download-button"])
                    download_analysis_file = gr.File(label="Analysis TXT Download", visible=False)
        
        with gr.Tab("Info"):
            create_info_panel()
        
        # Connect event handlers
        
        # Standard search - update chunks display
        search_components["standard_search_btn"].click(
            perform_retrieval_and_update_ui,
            inputs=[
                search_components["retrieval_query"], 
                search_components["chunk_size"],
                search_components["year_start"],
                search_components["year_end"],
                search_components["keywords"],
                search_components["search_in"],
                search_components["use_semantic_expansion"],
                search_components["semantic_expansion_factor"],
                search_components["expanded_words_state"],
                search_components["use_time_intervals"], 
                search_components["time_interval_size"], 
                search_components["top_k"],
                search_components["chunks_per_interval"]
            ],
            outputs=[
                search_components["search_status"],
                search_components["retrieved_chunks_state"],
                search_method_accordion,
                retrieved_texts_accordion,
                analysis_accordion
            ]
        ).then(
            lambda: gr.update(visible=False),  # Hide comprehensive download for standard
            outputs=[download_comprehensive_btn]
        ).then(
            # Update interactive chunks display
            update_chunks_display_handler,
            inputs=[search_components["retrieved_chunks_state"]],
            outputs=[
                chunks_display_components["chunks_selection_html"],
                chunks_display_components["selection_summary"],
                chunks_display_components["select_all_btn"],
                chunks_display_components["deselect_all_btn"],
                chunks_display_components["confirm_selection_btn"],  # NEW
                chunks_display_components["transfer_to_analysis_btn"],
                chunks_display_components["available_chunks_state"],
                chunks_display_components["confirmed_selection_state"]  # NEW - starts empty
            ]
        )

        # LLM-assisted search - update chunks display
        search_components["llm_assisted_search_btn"].click(
            perform_llm_assisted_search_threaded,
            inputs=[
                search_components["retrieval_query"],
                search_components["chunk_size"],
                search_components["year_start"],
                search_components["year_end"],
                search_components["llm_assisted_use_time_intervals"], 
                search_components["llm_assisted_time_interval_size"],  
                search_components["chunks_per_interval_initial"],  
                search_components["chunks_per_interval_final"],  
                search_components["llm_assisted_min_retrieval_score"],  
                search_components["llm_assisted_keywords"],  
                search_components["llm_assisted_search_in"],  
                gr.State(True),
                search_components["llm_assisted_model"], 
                search_components["llm_assisted_system_prompt_template"], 
                search_components["llm_assisted_system_prompt_text"] 
            ],
            outputs=[
                search_components["search_status"],
                search_components["retrieved_chunks_state"],
                search_components["search_mode"],
                search_components["llm_assisted_search_btn"],
                search_components["llm_assisted_cancel_btn"],
                search_components["llm_assisted_progress"],
                retrieved_texts_accordion,
                analysis_accordion
            ]
        ).then(
            lambda: gr.update(visible=True),  # Show comprehensive download for LLM-assisted
            outputs=[download_comprehensive_btn]
        ).then(
            # Update interactive chunks display for LLM-assisted
            update_chunks_display_handler,
            inputs=[search_components["retrieved_chunks_state"]],
            outputs=[
                chunks_display_components["chunks_selection_html"],
                chunks_display_components["selection_summary"],
                chunks_display_components["select_all_btn"],
                chunks_display_components["deselect_all_btn"],
                chunks_display_components["confirm_selection_btn"],  # NEW
                chunks_display_components["transfer_to_analysis_btn"],
                chunks_display_components["available_chunks_state"],
                chunks_display_components["confirmed_selection_state"]  # NEW
            ]
        )

        # Select all button - just updates summary text, JavaScript handles checkboxes
        chunks_display_components["select_all_btn"].click(
            handle_select_all,
            inputs=[chunks_display_components["available_chunks_state"]],
            outputs=[chunks_display_components["selection_summary"]]
        )
        
        # Deselect all button - just updates summary text, JavaScript handles checkboxes  
        chunks_display_components["deselect_all_btn"].click(
            handle_deselect_all,
            inputs=[chunks_display_components["available_chunks_state"]],
            outputs=[chunks_display_components["selection_summary"]]
        )

        # NEW: Confirmation button - just reads the hidden input state
        chunks_display_components["confirm_selection_btn"].click(
            confirm_selection,
            inputs=[
                chunks_display_components["js_selection_input"],
                chunks_display_components["available_chunks_state"]
            ],
            outputs=[
                chunks_display_components["confirmed_selection_state"],
                chunks_display_components["selection_summary"],
                chunks_display_components["transfer_to_analysis_btn"]
            ]
        )

        # FIXED: Transfer button - uses confirmed selection state
        chunks_display_components["transfer_to_analysis_btn"].click(
            transfer_chunks_to_analysis,
            inputs=[
                chunks_display_components["available_chunks_state"],
                chunks_display_components["confirmed_selection_state"]  # FIXED: Use confirmed state
            ],
            outputs=[
                chunks_display_components["transfer_status"],
                chunks_display_components["transferred_chunks_state"]
            ]
        ).then(
            # Update the analysis section with transferred chunks
            question_components["update_transferred_chunks_display"],
            inputs=[chunks_display_components["transferred_chunks_state"]],
            outputs=[
                question_components["transferred_chunks_display"],
                question_components["transferred_summary"],
                question_components["analyze_btn"]
            ]
        ).then(
            # Open analysis accordion and close retrieval accordion
            lambda: (gr.update(open=False), gr.update(open=True)),
            outputs=[retrieved_texts_accordion, analysis_accordion]
        )

        # LLM-assisted search cancellation
        search_components["llm_assisted_cancel_btn"].click(
            cancel_llm_assisted_search,
            outputs=[search_components["llm_assisted_progress"]]
        )
        
        
        # UPDATED: Analysis button click
        question_components["analyze_btn"].click(
            perform_analysis_and_update_ui_with_transferred_chunks, 
            inputs=[
                question_components["user_prompt"],
                question_components["transferred_chunks_state"],  # Use transferred chunks
                question_components["model_selection"],
                question_components["system_prompt_template"],
                question_components["system_prompt_text"],
                question_components["temperature"]
            ],
            outputs=[
                results_components["answer_output"],
                results_components["metadata_output"],
                analysis_accordion,
                results_accordion
            ]
        )


        # Download handlers (updated but keeping functionality)
        def handle_json_download(retrieved_chunks_state):
            """Handle JSON download with proper status updates."""
            try:
                file_path = create_download_json(retrieved_chunks_state)
                if file_path:
                    has_dual_scores = any(
                        'vector_similarity_score' in chunk or 'llm_evaluation_score' in chunk
                        for chunk in retrieved_chunks_state.get('chunks', [])
                    ) if retrieved_chunks_state else False
                    
                    summary = format_download_summary(
                        len(retrieved_chunks_state.get('chunks', [])), 
                        "JSON", 
                        has_dual_scores
                    )
                    return (
                        gr.update(value=summary, visible=True),
                        gr.update(value=file_path, visible=True)
                    )
                else:
                    return (
                        gr.update(value="❌ Fehler: Keine Daten zum Herunterladen verfügbar.", visible=True),
                        gr.update(visible=False)
                    )
            except Exception as e:
                logger.error(f"JSON download error: {e}")
                return (
                    gr.update(value=f"❌ Fehler beim Erstellen der JSON-Datei: {str(e)}", visible=True),
                    gr.update(visible=False)
                )
        
        def handle_csv_download(retrieved_chunks_state):
            """Handle CSV download with improved German text encoding."""
            try:
                file_path = create_download_csv(retrieved_chunks_state)
                if file_path:
                    has_dual_scores = any(
                        'vector_similarity_score' in chunk or 'llm_evaluation_score' in chunk
                        for chunk in retrieved_chunks_state.get('chunks', [])
                    ) if retrieved_chunks_state else False
                    
                    summary = format_download_summary(
                        len(retrieved_chunks_state.get('chunks', [])), 
                        "CSV", 
                        has_dual_scores
                    )
                    return (
                        gr.update(value=summary, visible=True),
                        gr.update(value=file_path, visible=True)
                    )
                else:
                    return (
                        gr.update(value="❌ Fehler: Keine Daten zum Herunterladen verfügbar.", visible=True),
                        gr.update(visible=False)
                    )
            except Exception as e:
                logger.error(f"CSV download error: {e}")
                return (
                    gr.update(value=f"❌ Fehler beim Erstellen der CSV-Datei: {str(e)}", visible=True),
                    gr.update(visible=False)
                )
        
        def handle_comprehensive_download(retrieved_chunks_state):
            """Handle comprehensive LLM-assisted download."""
            try:
                file_path = create_llm_assisted_download_comprehensive(retrieved_chunks_state)
                if file_path:
                    return (
                        gr.update(value="✅ Umfassende LLM-Datei wurde erstellt.", visible=True),
                        gr.update(value=file_path, visible=True)
                    )
                else:
                    return (
                        gr.update(value="❌ Fehler: Keine LLM-Daten zum Herunterladen verfügbar.", visible=True),
                        gr.update(visible=False)
                    )
            except Exception as e:
                logger.error(f"Comprehensive download error: {e}")
                return (
                    gr.update(value=f"❌ Fehler beim Erstellen der umfassenden Datei: {str(e)}", visible=True),
                    gr.update(visible=False)
                )
        
        # NEW: Analysis TXT download
        def handle_analysis_txt_download(answer_output, metadata_output):
            """Handle analysis results download as TXT."""
            try:
                import tempfile
                from datetime import datetime
                
                # Create TXT content
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                txt_content = f"""SPIEGEL RAG System - Analyse-Ergebnisse
Erstellt am: {timestamp}

{'='*60}
ANALYSE
{'='*60}

{answer_output}

{'='*60}
METADATEN
{'='*60}

{metadata_output}
"""
                
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w', 
                    suffix='.txt', 
                    prefix='spiegel_analysis_', 
                    delete=False,
                    encoding='utf-8'
                )
                
                temp_file.write(txt_content)
                temp_file.close()
                
                return (
                    gr.update(value="✅ Analyse als TXT-Datei erstellt.", visible=True),
                    gr.update(value=temp_file.name, visible=True)
                )
                
            except Exception as e:
                logger.error(f"Analysis TXT download error: {e}")
                return (
                    gr.update(value=f"❌ Fehler beim Erstellen der TXT-Datei: {str(e)}", visible=True),
                    gr.update(visible=False)
                )
        
        # Connect download events
        download_json_btn.click(
            handle_json_download,
            inputs=[search_components["retrieved_chunks_state"]],
            outputs=[download_status, download_json_file]
        )
        
        download_csv_btn.click(
            handle_csv_download,
            inputs=[search_components["retrieved_chunks_state"]],
            outputs=[download_status, download_csv_file]
        )
        
        download_comprehensive_btn.click(
            handle_comprehensive_download,
            inputs=[search_components["retrieved_chunks_state"]],
            outputs=[download_status, download_comprehensive_file]
        )
        
        # NEW: Analysis TXT download
        download_analysis_btn.click(
            handle_analysis_txt_download,
            inputs=[results_components["answer_output"], results_components["metadata_output"]],
            outputs=[download_status, download_analysis_file]
        )
        
        logger.info("Gradio interface created successfully with updated terminology and design")

    return app

def main():
    """Main entry point for the updated application."""
    try:
        app = create_app()
        logger.info("Starting updated Gradio application with new terminology and design...")
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    main()