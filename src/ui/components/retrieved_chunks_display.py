# src/ui/components/retrieved_chunks_display_enhanced.py
"""
ENHANCED NATIVE SOLUTION: Combines native CheckboxGroup selection with full content display.
Handles large numbers of chunks (150+) efficiently while showing all metadata and content.
"""
import gradio as gr
from typing import Dict, List, Any, Optional, Tuple
import math

def create_enhanced_chunks_display() -> Dict[str, Any]:
    """Create enhanced chunks display with native selection + full content view."""
    
    with gr.Group(elem_classes=["form-container"]):
        gr.HTML("<h3 style='margin-top: 0; color: var(--text-primary);'>📄 Gefundene Texte</h3>")
        
        # Status and summary
        chunks_status = gr.Markdown("**Noch keine Texte verfügbar**")
        
        # Detailed content display with pagination for large datasets
        gr.Markdown("#### Vollständige Textansicht")
        
        # PROMINENT Pagination controls for large datasets
        with gr.Row():
            pagination_info = gr.Markdown("", visible=False)
        with gr.Row():
            prev_page_btn = gr.Button("⬅️ Vorherige Seite", size="sm", visible=False, elem_classes=["btn-secondary"])
            page_selector = gr.Dropdown(
                choices=[],
                value=None,
                label="Springe zu Seite",
                visible=False,
                interactive=True,
                scale=2
            )
            next_page_btn = gr.Button("Nächste Seite ➡️", size="sm", visible=False, elem_classes=["btn-secondary"])
        
        # Full content display (paginated for performance)
        chunks_content_display = gr.HTML(
            value="<div class='info-message'><p><em>Führen Sie zuerst eine Heuristik durch...</em></p></div>",
            elem_classes=["chunks-content-display"]
        )
        
        # MOVED: Selection interface BELOW the content display
        gr.Markdown("---")
        gr.Markdown("#### Auswahl für Analyse")
        
        # CORE: Native CheckboxGroup for selection (handles 150+ items efficiently)
        chunks_selector = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="Wählen Sie die Texte aus, die zur Analyse übertragen werden sollen:",
            visible=False,
            interactive=True,
            elem_classes=["chunks-selector"]
        )
        
        # Control buttons with clear explanations
        with gr.Row():
            select_all_btn = gr.Button("✅ Alle auswählen", size="sm", visible=False)
            deselect_all_btn = gr.Button("❌ Alle abwählen", size="sm", visible=False)
            invert_selection_btn = gr.Button("🔄 Auswahl umkehren", size="sm", visible=False)
        
        # Selection summary
        selection_summary = gr.Markdown("", visible=False)
        
        # Filter options for large datasets
        with gr.Accordion("Anzeige-Optionen", open=False, visible=False) as display_options:
            with gr.Row():
                show_selected_only = gr.Checkbox(
                    label="Nur ausgewählte anzeigen",
                    value=False
                )
                chunks_per_page = gr.Dropdown(
                    choices=[10, 25, 50, 100],
                    value=25,
                    label="Texte pro Seite"
                )
        
        # Transfer section
        with gr.Row():
            transfer_btn = gr.Button(
                "🔄 Ausgewählte Quellen zur Analyse übertragen",
                variant="primary",
                visible=False
            )
        
        transfer_status = gr.Markdown(value="", visible=False)
        
        # State management
        available_chunks_state = gr.State([])
        transferred_chunks_state = gr.State([])
        current_page_state = gr.State(1)
        chunks_per_page_state = gr.State(25)
    
    return {
        "chunks_status": chunks_status,
        "chunks_selector": chunks_selector,
        "chunks_content_display": chunks_content_display,
        "select_all_btn": select_all_btn,
        "deselect_all_btn": deselect_all_btn,
        "invert_selection_btn": invert_selection_btn,
        "selection_summary": selection_summary,
        "transfer_btn": transfer_btn,
        "transfer_status": transfer_status,
        "available_chunks_state": available_chunks_state,
        "transferred_chunks_state": transferred_chunks_state,
        
        # Pagination components
        "pagination_info": pagination_info,
        "prev_page_btn": prev_page_btn,
        "page_selector": page_selector,
        "next_page_btn": next_page_btn,
        "current_page_state": current_page_state,
        "chunks_per_page_state": chunks_per_page_state,
        
        # Display options
        "display_options": display_options,
        "show_selected_only": show_selected_only,
        "chunks_per_page": chunks_per_page,
    }

def update_chunks_display_enhanced(retrieved_chunks: Dict[str, Any]) -> Tuple:
    """Update display when new chunks are retrieved - optimized for large datasets."""
    
    if not retrieved_chunks or not retrieved_chunks.get('chunks'):
        return (
            "**Noch keine Texte verfügbar**",  # chunks_status
            gr.update(choices=[], value=[], visible=False),  # chunks_selector
            "<div class='info-message'><p><em>Keine Texte verfügbar.</em></p></div>",  # chunks_content_display
            gr.update(visible=False),  # select_all_btn
            gr.update(visible=False),  # deselect_all_btn
            gr.update(visible=False),  # invert_selection_btn
            "",  # selection_summary
            gr.update(visible=False),  # transfer_btn
            [],  # available_chunks_state
            gr.update(visible=False),  # pagination_info
            gr.update(visible=False),  # prev_page_btn
            gr.update(choices=[], value=None, visible=False),  # page_selector
            gr.update(visible=False),  # next_page_btn
            gr.update(visible=False),  # display_options
            1,  # current_page_state
            25  # chunks_per_page_state
        )
    
    chunks = retrieved_chunks.get('chunks', [])
    total_chunks = len(chunks)
    
    # Create choices for CheckboxGroup - optimized format
    choices = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get('metadata', {}).get('Artikeltitel', 'Kein Titel')
        score = chunk.get('relevance_score', 0.0)
        
        # Truncate long titles for CheckboxGroup performance
        if len(title) > 50:
            title = title[:47] + "..."
        
        # Format: "1. Title (Score: 0.123)"
        choice_label = f"{i}. {title} (Score: {score:.3f})"
        choices.append(choice_label)
    
    # Default: select all chunks
    selected_choices = choices.copy()
    
    # Determine if pagination is needed (for 25+ chunks instead of 50+ for better UX)
    use_pagination = total_chunks > 25
    chunks_per_page = 25
    total_pages = math.ceil(total_chunks / chunks_per_page) if use_pagination else 1
    
    # Create initial content display (first page)
    content_html = create_paginated_content_display(
        chunks, 
        page=1, 
        chunks_per_page=chunks_per_page,
        selected_indices=list(range(1, total_chunks + 1)),  # All selected initially
        show_selected_only=False
    )
    
    # Status message
    search_method = retrieved_chunks.get('metadata', {}).get('retrieval_method', 'standard')
    method_display = "LLM-Unterstützte Auswahl" if 'llm_assisted' in search_method else "Standard-Heuristik"
    
    status_text = f"**Verfügbare Texte**: {total_chunks} ({method_display})"
    if use_pagination:
        status_text += f" | **Seitenweise Anzeige**: {chunks_per_page} Texte pro Seite"
    
    # Pagination setup - IMPROVED VISIBILITY
    page_choices = [f"Seite {i}" for i in range(1, total_pages + 1)] if use_pagination else []
    pagination_text = f"**📄 Seite 1 von {total_pages}** | Texte 1-{min(chunks_per_page, total_chunks)} von {total_chunks}" if use_pagination else ""
    
    # Selection summary
    selection_text = f"**Ausgewählt**: {total_chunks} von {total_chunks} (alle)"
    
    return (
        status_text,  # chunks_status
        gr.update(choices=choices, value=selected_choices, visible=True),  # chunks_selector
        content_html,  # chunks_content_display
        gr.update(visible=True),  # select_all_btn
        gr.update(visible=True),  # deselect_all_btn
        gr.update(visible=True),  # invert_selection_btn
        selection_text,  # selection_summary
        gr.update(visible=True),  # transfer_btn
        chunks,  # available_chunks_state
        gr.update(value=pagination_text, visible=use_pagination),  # pagination_info
        gr.update(visible=use_pagination),  # prev_page_btn - ALWAYS show if pagination needed
        gr.update(choices=page_choices, value="Seite 1" if use_pagination else None, visible=use_pagination),  # page_selector
        gr.update(visible=use_pagination),  # next_page_btn - ALWAYS show if pagination needed
        gr.update(visible=use_pagination),  # display_options
        1,  # current_page_state
        chunks_per_page  # chunks_per_page_state
    )

def create_paginated_content_display(
    chunks: List[Dict], 
    page: int = 1, 
    chunks_per_page: int = 25,
    selected_indices: List[int] = None,
    show_selected_only: bool = False
) -> str:
    """Create paginated HTML display with full content, metadata, and scores."""
    
    if not chunks:
        return "<div class='info-message'><p><em>Keine Texte verfügbar.</em></p></div>"
    
    # Filter chunks if showing selected only
    display_chunks = []
    if show_selected_only and selected_indices:
        for idx in selected_indices:
            if 1 <= idx <= len(chunks):
                chunk = chunks[idx - 1].copy()
                chunk['display_id'] = idx
                display_chunks.append(chunk)
    else:
        display_chunks = [chunk.copy() for chunk in chunks]
        for i, chunk in enumerate(display_chunks):
            chunk['display_id'] = i + 1
    
    total_display_chunks = len(display_chunks)
    
    # Calculate pagination
    start_idx = (page - 1) * chunks_per_page
    end_idx = min(start_idx + chunks_per_page, total_display_chunks)
    page_chunks = display_chunks[start_idx:end_idx]
    
    if not page_chunks:
        return "<div class='info-message'><p><em>Keine Texte auf dieser Seite.</em></p></div>"
    
    # Create performance-optimized HTML
    html_content = f"""
    <div style='max-height: 80vh; overflow-y: auto; padding: 10px;'>
        <div class="success-message" style="margin-bottom: 15px;">
            <h4 style="color: var(--text-primary); margin-top: 0;">
                📄 Angezeigt: Texte {start_idx + 1}-{end_idx} von {total_display_chunks}
            </h4>
            <p style="color: var(--text-secondary); margin: 5px 0;">
                Verwenden Sie die Navigationsschaltflächen oben, um zwischen den Seiten zu wechseln.
            </p>
        </div>
    """
    
    for chunk in page_chunks:
        display_id = chunk.get('display_id', '?')
        metadata = chunk.get('metadata', {})
        content = chunk.get('content', '')
        relevance_score = chunk.get('relevance_score', 0.0)
        
        # Get additional scores if available (LLM-assisted search)
        vector_score = chunk.get('vector_similarity_score', relevance_score)
        llm_score = chunk.get('llm_evaluation_score', None)
        
        # Metadata extraction
        title = metadata.get('Artikeltitel', 'Kein Titel')
        date = metadata.get('Datum', 'Unbekannt')
        year = metadata.get('Jahrgang', 'Unbekannt')
        url = metadata.get('URL', '')
        authors = metadata.get('Autoren', '')
        
        # Selection indicator
        is_selected = selected_indices and display_id in selected_indices
        selection_indicator = "✅" if is_selected else "◻️"
        
        html_content += f"""
        <div style="
            background: var(--bg-tertiary); 
            border: 1px solid var(--border-primary); 
            border-radius: 8px; 
            padding: 15px; 
            margin-bottom: 15px;
            border-left: 4px solid {'var(--brand-primary)' if is_selected else 'var(--brand-secondary)'};
        ">
            <!-- Header with selection status -->
            <div style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 12px;">
                <span style="font-size: 18px;">{selection_indicator}</span>
                <div style="flex: 1;">
                    <div style="color: var(--text-primary); font-weight: 600; font-size: 16px; margin-bottom: 6px;">
                        {display_id}. {title}
                    </div>
                    <div style="color: var(--text-secondary); font-size: 14px; margin-bottom: 8px;">
                        <strong>Datum:</strong> {date} | 
                        <strong>Jahr:</strong> {year}"""
        
        # Add scores - if dual scores are available, only show LLM and Vector, otherwise show Relevanz
        if llm_score is not None:
            html_content += f""" | 
                        <span style="color: var(--brand-accent);">
                            <strong>LLM:</strong> {llm_score:.3f} | 
                            <strong>Vector:</strong> {vector_score:.3f}
                        </span>"""
        else:
            html_content += f""" | 
                        <strong>Relevanz:</strong> {relevance_score:.3f}"""
        
        # Add author info if available
        if authors:
            html_content += f"""<br>
                        <strong>Autoren:</strong> {authors}"""
        
        # Add URL if available
        if url and url != 'Keine URL':
            html_content += f"""<br>
                        <a href="{url}" target="_blank" style="color: var(--brand-primary); text-decoration: none;">
                            🔗 Artikel öffnen
                        </a>"""
        
        html_content += """
                    </div>
                </div>
            </div>
        """
        
        # Show LLM evaluation reasoning (Begründung) if available
        evaluation_text = metadata.get('evaluation_text', '')
        if evaluation_text:
            # Extract reasoning from evaluation text using improved parsing
            reasoning = ""
            if '**Argumentation:**' in evaluation_text:
                # Extract text after "Argumentation:" 
                reasoning = evaluation_text.split('**Argumentation:**', 1)[1].strip()
                # Remove any score information at the end
                if 'Score:' in reasoning:
                    reasoning = reasoning.split('Score:')[0].strip()
            elif '-' in evaluation_text:
                # Original dash separator format
                reasoning = evaluation_text.split('-', 1)[1].strip()
            else:
                # Use full evaluation text as fallback, but clean it up
                reasoning = evaluation_text.strip()
                # Remove common prefixes that aren't part of reasoning
                if reasoning.startswith('Text ') and ':' in reasoning:
                    reasoning = reasoning.split(':', 1)[1].strip()
            
            if reasoning and reasoning != "Automatisch extrahiert" and len(reasoning) > 10:
                html_content += f"""
            <div style="
                background: var(--bg-primary); 
                border-left: 3px solid var(--brand-accent); 
                padding: 10px; 
                border-radius: 4px; 
                margin-bottom: 12px;
                color: var(--text-secondary);
            ">
                <strong style="color: var(--brand-accent);">💭 Begründung:</strong> {reasoning}
            </div>
            """
        
        # Full text content with collapsible option for very long texts
        content_length = len(content)
        if content_length > 2000:
            # For very long texts, show preview + expandable
            content_preview = content[:500] + "..."
            html_content += f"""
            <details style="margin-top: 10px;">
                <summary style="
                    color: var(--text-primary); 
                    font-weight: 500; 
                    cursor: pointer; 
                    padding: 8px 0;
                    border-bottom: 1px solid var(--border-primary);
                ">
                    📄 Volltext anzeigen ({content_length:,} Zeichen)
                </summary>
                <div style="
                    background: var(--bg-primary); 
                    border-left: 3px solid var(--brand-secondary); 
                    padding: 15px; 
                    border-radius: 4px; 
                    margin-top: 10px;
                    color: var(--text-secondary);
                    line-height: 1.6;
                    white-space: pre-wrap;
                    max-height: 400px;
                    overflow-y: auto;
                ">
                    {content}
                </div>
            </details>
            """
        else:
            # For shorter texts, show directly
            html_content += f"""
            <div style="
                background: var(--bg-primary); 
                border-left: 3px solid var(--brand-secondary); 
                padding: 15px; 
                border-radius: 4px; 
                margin-top: 10px;
                color: var(--text-secondary);
                line-height: 1.6;
                white-space: pre-wrap;
            ">
                <strong style="color: var(--text-primary);">Volltext:</strong><br><br>
                {content}
            </div>
            """
        
        html_content += "</div>"
    
    html_content += "</div>"
    return html_content

def update_selection_and_display(
    selected_choices: List[str],
    available_chunks: List[Dict],
    current_page: int,
    chunks_per_page: int,
    show_selected_only: bool
) -> Tuple[str, str, gr.update]:
    """Update selection summary and content display when selection changes."""
    
    if not available_chunks:
        return "**Keine Texte verfügbar**", "", gr.update(visible=False)
    
    total_chunks = len(available_chunks)
    selected_count = len(selected_choices)
    
    # Extract selected indices from choices
    selected_indices = []
    for choice in selected_choices:
        try:
            chunk_id = int(choice.split('.')[0])
            selected_indices.append(chunk_id)
        except (ValueError, IndexError):
            continue
    
    # Update content display
    content_html = create_paginated_content_display(
        available_chunks,
        page=current_page,
        chunks_per_page=chunks_per_page,
        selected_indices=selected_indices,
        show_selected_only=show_selected_only
    )
    
    # Update selection summary
    if selected_count == total_chunks:
        selection_text = f"**Ausgewählt**: {selected_count} von {total_chunks} (alle)"
    elif selected_count == 0:
        selection_text = f"**Ausgewählt**: 0 von {total_chunks} (keine)"
    else:
        percentage = (selected_count / total_chunks) * 100
        selection_text = f"**Ausgewählt**: {selected_count} von {total_chunks} ({percentage:.0f}%)"
    
    # Show/hide transfer button
    transfer_btn_state = gr.update(visible=selected_count > 0)
    
    return selection_text, content_html, transfer_btn_state

def handle_page_navigation(
    direction: str,
    current_page: int,
    available_chunks: List[Dict],
    chunks_per_page: int,
    selected_choices: List[str],
    show_selected_only: bool,
    goto_page: int = None
) -> Tuple[str, str, int]:
    """Handle pagination navigation."""
    
    if not available_chunks:
        return "", "Keine Texte verfügbar", current_page
    
    total_chunks = len(available_chunks)
    total_pages = math.ceil(total_chunks / chunks_per_page)
    
    # Calculate new page
    if direction == "next":
        new_page = min(current_page + 1, total_pages)
    elif direction == "prev":
        new_page = max(current_page - 1, 1)
    elif direction == "goto" and goto_page:
        new_page = max(1, min(goto_page, total_pages))
    else:
        new_page = current_page
    
    # Extract selected indices
    selected_indices = []
    for choice in selected_choices:
        try:
            chunk_id = int(choice.split('.')[0])
            selected_indices.append(chunk_id)
        except (ValueError, IndexError):
            continue
    
    # Update content display
    content_html = create_paginated_content_display(
        available_chunks,
        page=new_page,
        chunks_per_page=chunks_per_page,
        selected_indices=selected_indices,
        show_selected_only=show_selected_only
    )
    
    # Update pagination info
    start_idx = (new_page - 1) * chunks_per_page + 1
    end_idx = min(new_page * chunks_per_page, total_chunks)
    pagination_text = f"**📄 Seite {new_page} von {total_pages}** | Texte {start_idx}-{end_idx} von {total_chunks}"
    
    return pagination_text, content_html, new_page

# Simple selection handlers
def handle_select_all_enhanced(all_choices: List[str]) -> List[str]:
    """Select all chunks."""
    return all_choices

def handle_deselect_all_enhanced() -> List[str]:
    """Deselect all chunks."""
    return []

def handle_invert_selection_enhanced(selected_choices: List[str], all_choices: List[str]) -> List[str]:
    """Invert current selection."""
    return [choice for choice in all_choices if choice not in selected_choices]

def transfer_chunks_enhanced(
    selected_choices: List[str], 
    available_chunks: List[Dict]
) -> Tuple[gr.update, List[Dict]]:
    """Transfer selected chunks to analysis."""
    
    if not available_chunks:
        error_message = """<div class="error-message">
        <h4>❌ Keine Texte verfügbar</h4>
        <p>Führen Sie zuerst eine Heuristik durch.</p>
        </div>"""
        return gr.update(value=error_message, visible=True), []
    
    if not selected_choices:
        error_message = """<div class="error-message">
        <h4>❌ Keine Texte ausgewählt</h4>
        <p>Wählen Sie mindestens einen Text aus.</p>
        </div>"""
        return gr.update(value=error_message, visible=True), []
    
    # Extract chunk IDs from choices
    selected_ids = []
    for choice in selected_choices:
        try:
            chunk_id = int(choice.split('.')[0])
            selected_ids.append(chunk_id)
        except (ValueError, IndexError):
            continue
    
    # Transfer selected chunks
    transferred_chunks = []
    for chunk_id in sorted(selected_ids):
        index = chunk_id - 1
        if 0 <= index < len(available_chunks):
            chunk = available_chunks[index].copy()
            chunk['transferred_id'] = chunk_id
            transferred_chunks.append(chunk)
    
    if not transferred_chunks:
        error_message = """<div class="error-message">
        <h4>❌ Übertragung fehlgeschlagen</h4>
        <p>Keine gültigen Texte in der Auswahl gefunden.</p>
        </div>"""
        return gr.update(value=error_message, visible=True), []
    
    success_message = f"""<div class="success-message">
    <h4>✅ Texte erfolgreich übertragen</h4>
    <p><strong>{len(transferred_chunks)} von {len(available_chunks)} Texten</strong> wurden zur Analyse übertragen.</p>
    
    <p><strong>Übertragene Text-IDs:</strong> {', '.join(map(str, sorted(selected_ids[:10])))}</p>
    {f'<p><em>...und {len(selected_ids) - 10} weitere</em></p>' if len(selected_ids) > 10 else ''}
    
    <p><em>Sie können jederzeit eine neue Auswahl treffen und erneut übertragen.</em></p>
    </div>"""
    
    return gr.update(value=success_message, visible=True), transferred_chunks