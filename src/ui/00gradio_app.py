"""
Gradio interface for Spiegel RAG.
"""
import os
import sys
import logging
import time
from typing import Dict, List, Tuple, Optional

import gradio as gr

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.rag_engine import SpiegelRAGEngine
from src.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize RAG engine
rag_engine = SpiegelRAGEngine()


def perform_search(
    query: str,
    question: str,
    chunk_size: int,
    year_start: int,
    year_end: int,
    use_query_refinement: bool,
    with_citations: bool,
    disable_filters: bool,
    system_prompt_selection: str,
    custom_system_prompt: str,
    keywords: str, 
    search_in: List[str],
    use_semantic_expansion: bool,
    semantic_expansion_factor: int,
    enforce_keywords: bool
) -> Tuple[str, str, str]:

    """
    Perform search with custom system prompt support.
    """
    try:
        # Start timing
        start_time = time.time()
        logger.info(f"Search started with query: '{query}', question: '{question}'")

        # Determine which system prompt to use
        system_prompt = None
        
        # If "Custom" is selected and custom text is provided, use that
        if system_prompt_selection == "Custom" and custom_system_prompt.strip():
            system_prompt = custom_system_prompt
        # Otherwise use the predefined prompt
        elif system_prompt_selection in settings.SYSTEM_PROMPTS:
            system_prompt = settings.SYSTEM_PROMPTS[system_prompt_selection]
        
        # Set year_range to None if filters are disabled
        year_range = None if disable_filters else [year_start, year_end]
        
        # Process keywords if provided
        keywords_to_use = keywords.strip() if keywords and keywords.strip() else None
        
        logger.info(f"Starting RAG search with chunk_size={chunk_size}, year_range={year_range}, use_query_refinement={use_query_refinement}")

        # Perform search
        results = rag_engine.search(
            question=question,
            content_description=query,
            year_range=year_range,
            chunk_size=chunk_size,
            keywords=keywords_to_use,
            search_in=search_in if search_in else None,
            use_query_refinement=use_query_refinement,
            with_citations=with_citations,
            system_prompt=system_prompt,
            use_semantic_expansion=use_semantic_expansion,
            semantic_expansion_factor=semantic_expansion_factor,
            enforce_keywords=enforce_keywords
        )

        logger.info(f"Search completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Found {len(results.get('chunks', []))} chunks")
        
        # Format retrieved chunks for display
        chunks_formatted = ""
        if results["chunks"]:
            for i, chunk in enumerate(results["chunks"], 1):
                metadata = chunk["metadata"]
                chunks_formatted += f"### {i}. {metadata.get('Artikeltitel', 'Kein Titel')}\n"
                chunks_formatted += f"**Datum:** {metadata.get('Datum', 'Unbekannt')} | "
                chunks_formatted += f"**Relevanz:** {chunk['relevance_score']:.2f}\n\n"
                chunks_formatted += f"{chunk['content']}\n\n"
                chunks_formatted += "---\n\n"
        else:
            chunks_formatted = "Keine Chunks gefunden."
        
        metadata_str = f"### Search Metadata\n\n"
        metadata_str += f"- **Query:** {results['metadata']['query']}\n"
        metadata_str += f"- **Chunk Size:** {results['metadata']['chunk_size']}\n"
        metadata_str += f"- **Year Range:** {results['metadata']['year_range']}\n"

        # Format metadata
        metadata_str += f"- **System Prompt:** "
        if system_prompt_selection == "Custom":
            metadata_str += "Custom (user-defined)\n"
        else:
            metadata_str += f"{system_prompt_selection}\n"
        
        if keywords_to_use:
            metadata_str += f"- **Keywords:** {keywords_to_use}\n"
        
        if search_in:
            metadata_str += f"- **Search In:** {', '.join(search_in)}\n"
            
        if use_semantic_expansion and keywords_to_use:
            metadata_str += f"- **Semantic Expansion:** Enabled (factor: {semantic_expansion_factor})\n"
        
        if 'model' in results['metadata']:
            metadata_str += f"- **Model:** {results['metadata']['model']}\n"
        
        # Add refined queries if available
        if 'refined_queries' in results['metadata'] and results['metadata']['refined_queries']:
            metadata_str += f"\n### Refined Queries\n\n"
            for i, q in enumerate(results['metadata']['refined_queries'], 1):
                metadata_str += f"{i}. {q}\n"
        
        # Add expanded keywords if available
        if 'expanded_keywords' in results.get('metadata', {}):
            metadata_str += f"\n### Expanded Keywords\n\n"
            expanded = results['metadata']['expanded_keywords']
            for category, terms in expanded.items():
                metadata_str += f"**{category.capitalize()}:** "
                term_list = []
                for term in terms:
                    if isinstance(term, dict) and 'original' in term:
                        original = term['original']
                        expanded_terms = [f"{w['word']} ({w['similarity']:.2f})" 
                                         for w in term.get('expanded', {}).get(original, [])]
                        term_str = f"{original} → {', '.join(expanded_terms)}" if expanded_terms else original
                        term_list.append(term_str)
                metadata_str += ", ".join(term_list) + "\n"
        
        # Add citations if available
        if "citations" in results:
            metadata_str += f"\n### Citations\n\n"
            for citation in results["citations"]:
                metadata_str += f"{citation}\n"
        
        # Add errors if available
        if 'error' in results.get('metadata', {}):
            metadata_str += f"\n### Errors\n\n"
            metadata_str += f"Error: {results['metadata']['error']}\n"
        
        logger.info("Results formatted successfully, returning to UI")

        return results["answer"], chunks_formatted, metadata_str
    except Exception as e:
        logger.error(f"Error in search: {e}", exc_info=True)
        return f"Error: {str(e)}", "", f"### Error\n\n{str(e)}"

def expand_keywords(keyword_expression: str, expansion_factor: int) -> str:
    """Find semantically similar words for a keyword expression."""
    if not keyword_expression.strip():
        return "Please enter keywords to expand."
    
    try:
        # Parse boolean expression
        parsed_terms = rag_engine.embedding_service.parse_boolean_expression(keyword_expression)
        
        # Expand terms with semantically similar words
        expanded_terms = rag_engine.embedding_service.filter_by_semantic_similarity(
            parsed_terms, 
            expansion_factor=expansion_factor
        )
        
        # Format the expanded terms for display
        result = f"## Expanded Keywords for: {keyword_expression}\n\n"
        
        for category, terms in expanded_terms.items():
            if terms:
                result += f"### {category.capitalize()} Terms:\n\n"
                for term_data in terms:
                    original = term_data.get('original', '')
                    expanded = term_data.get('expanded', {}).get(original, [])
                    result += f"**{original}**: "
                    if expanded:
                        result += ", ".join([f"{item['word']} ({item['similarity']:.2f})" for item in expanded])
                    else:
                        result += "No similar words found."
                    result += "\n\n"
        
        return result
    except Exception as e:
        logger.error(f"Error expanding keywords: {e}", exc_info=True)
        return f"Error expanding keywords: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Der Spiegel RAG (1948-1979)") as app:
    gr.Markdown("# Der Spiegel Archiv RAG (1948-1979)")
    gr.Markdown("Durchsuchen und analysieren Sie das Spiegel-Archiv mit Hilfe von Retrieval Augmented Generation.")
    
    with gr.Tabs():
        with gr.TabItem("Search & Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Search inputs
                    query = gr.Textbox(
                        label="Search Query (what to retrieve)",
                        placeholder="Beschreiben Sie, was Sie im Archiv finden möchten...",
                        lines=2
                    )
                    
                    question = gr.Textbox(
                        label="Question (what to answer)",
                        placeholder="Was möchten Sie über die gefundenen Inhalte wissen?",
                        lines=2
                    )
                    
                    # Keyword filtering with hard filter option
                    with gr.Accordion("Advanced Keyword Filtering", open=False):
                        keywords = gr.Textbox(
                            label="Boolean Keyword Expression",
                            placeholder="berlin AND (mauer OR wall) NOT soviet",
                            lines=2
                        )
                        
                        search_in = gr.CheckboxGroup(
                            choices=["Text", "Artikeltitel", "Schlagworte"],
                            value=["Text"],
                            label="Search In"
                        )
                        
                        with gr.Row():
                            use_semantic_expansion = gr.Checkbox(
                                label="Use Semantic Expansion",
                                value=True
                            )
                            
                            semantic_expansion_factor = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=3,
                                step=1,
                                label="Expansion Factor"
                            )
                            
                            enforce_keywords = gr.Checkbox(
                                label="Strict Keyword Filtering",
                                value=True,
                                info="Only return results that contain the keywords"
                            )
                    
                    with gr.Row():
                        chunk_size = gr.Dropdown(
                            choices=[300, 600, 1200, 2400, 3000],
                            value=3000,
                            label="Chunk Size"
                        )
                    
                    with gr.Row():
                        year_start = gr.Slider(
                            minimum=settings.MIN_YEAR,
                            maximum=settings.MAX_YEAR,
                            value=settings.MIN_YEAR,
                            step=1,
                            label="Start Year"
                        )
                        
                        year_end = gr.Slider(
                            minimum=settings.MIN_YEAR,
                            maximum=settings.MAX_YEAR,
                            value=settings.MAX_YEAR,
                            step=1,
                            label="End Year"
                        )
                    
                    with gr.Row():
                        use_query_refinement = gr.Checkbox(
                            label="Use Query Refinement",
                            value=True
                        )
                        
                        with_citations = gr.Checkbox(
                            label="Include Citations",
                            value=True
                        )
                        
                        disable_filters = gr.Checkbox(
                            label="Disable Year Filters (Debug)",
                            value=False
                        )
                    
                    # Custom system prompt section
                    with gr.Accordion("System Prompt Configuration", open=True):
                        system_prompt_selection = gr.Dropdown(
                            choices=list(settings.SYSTEM_PROMPTS.keys()) + ["Custom"],
                            value="with_citations",
                            label="System Prompt Template"
                        )
                        
                        custom_system_prompt = gr.Textbox(
                            label="Custom System Prompt",
                            placeholder="Enter your custom system prompt here...",
                            lines=6,
                            visible=False  # Initially hidden
                        )
                        
                        # Show example of the selected prompt
                        prompt_preview = gr.Markdown(
                            value=f"**Current Prompt:**\n\n```\n{settings.SYSTEM_PROMPTS['with_citations']}\n```"
                        )
                    
                    search_btn = gr.Button("Search and Generate Answer", variant="primary")
                
                with gr.Column(scale=1):
                    # Results (no changes here)
                    answer_output = gr.Markdown(
                        label="Answer",
                        value=""
                    )
                    
                    with gr.Tabs():
                        with gr.TabItem("Retrieved Chunks"):
                            chunks_output = gr.Markdown(
                                value=""
                            )
                        with gr.TabItem("Metadata"):
                            metadata_output = gr.Markdown(
                                value=""
                            )
    
  
    # Function to toggle custom prompt visibility and update preview
    def update_prompt_section(selection):
        if selection == "Custom":
            return gr.update(visible=True), ""
        else:
            prompt_text = settings.SYSTEM_PROMPTS.get(selection, "")
            preview = f"**Current Prompt:**\n\n```\n{prompt_text}\n```"
            return gr.update(visible=False), preview
    
    # Set up event handlers
    system_prompt_selection.change(
        update_prompt_section,
        inputs=[system_prompt_selection],
        outputs=[custom_system_prompt, prompt_preview]
    )
    
    search_btn.click(
        perform_search,
        inputs=[
            query,
            question,
            chunk_size,
            year_start,
            year_end,
            use_query_refinement,
            with_citations,
            disable_filters,
            system_prompt_selection,
            custom_system_prompt,
            keywords,
            search_in,
            use_semantic_expansion,
            semantic_expansion_factor,
            enforce_keywords
        ],
        outputs=[answer_output, chunks_output, metadata_output]
    )


# Run the app
if __name__ == "__main__":
    app.launch(share=False)