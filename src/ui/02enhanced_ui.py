# src/ui/enhanced_ui.py
import os
import sys
import logging
import json
from typing import Dict, List, Tuple, Optional
import time

import gradio as gr

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.rag_engine import SpiegelRAGEngine
from src.core.embedding_service import WordEmbeddingService
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize RAG engine and embedding service
try:
    logger.info("Initializing RAG Engine...")
    rag_engine = SpiegelRAGEngine()
    embedding_service = rag_engine.embedding_service
    logger.info("RAG Engine and Embedding Service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG Engine: {e}")
    rag_engine = None
    embedding_service = None

def find_similar_words(keyword: str, expansion_factor: int) -> str:
    """Find similar words for a given keyword using FastText embeddings."""
    if not keyword.strip():
        return "Please enter a keyword."
    
    if not embedding_service:
        return "Embedding service not available."
    
    try:
        # Get similar words
        similar_words = embedding_service.find_similar_words(keyword.strip(), top_n=expansion_factor)
        
        # Format results for display
        if not similar_words:
            return f"No similar words found for '{keyword}'"
        
        result = f"### Similar words for '{keyword}':\n\n"
        for word_info in similar_words:
            result += f"- **{word_info['word']}** (similarity: {word_info['similarity']:.4f})\n"
        
        return result
    except Exception as e:
        logger.error(f"Error finding similar words: {e}")
        return f"Error finding similar words: {str(e)}"

def parse_boolean_expression(expression: str) -> Dict[str, List[str]]:
    """Parse a boolean expression and display the parsed structure."""
    if not expression.strip():
        return "Please enter a boolean expression."
    
    if not embedding_service:
        return "Embedding service not available."
    
    try:
        # Parse the expression
        parsed = embedding_service.parse_boolean_expression(expression)
        
        # Format the result
        result = "### Parsed Boolean Expression:\n\n"
        result += f"Original: '{expression}'\n\n"
        
        if parsed["must"]:
            result += f"**MUST include all of**: {', '.join(parsed['must'])}\n\n"
        
        if parsed["should"]:
            result += f"**SHOULD include at least one of**: {', '.join(parsed['should'])}\n\n"
        
        if parsed["must_not"]:
            result += f"**MUST NOT include any of**: {', '.join(parsed['must_not'])}\n\n"
        
        return result
    except Exception as e:
        logger.error(f"Error parsing boolean expression: {e}")
        return f"Error parsing expression: {str(e)}"

def expand_boolean_expression(expression: str, expansion_factor: int) -> Tuple[str, str]:
    """Expand a boolean expression with semantically similar words."""
    if not expression.strip():
        return "Please enter a boolean expression.", ""
    
    if not embedding_service:
        return "Embedding service not available.", ""
    
    try:
        # Parse the boolean expression
        parsed_terms = embedding_service.parse_boolean_expression(expression)
        
        # Expand terms with semantically similar words
        expanded_terms = embedding_service.filter_by_semantic_similarity(
            parsed_terms, 
            expansion_factor=expansion_factor
        )
        
        # Format the expanded terms for display
        display_result = f"## Expanded Boolean Expression\n\n"
        display_result += f"Original expression: `{expression}`\n\n"
        
        # Save expanded words for potential use in search
        expanded_words = {}
        
        for category, terms in expanded_terms.items():
            if terms:
                display_result += f"### {category.capitalize()} Terms:\n\n"
                for term_data in terms:
                    original = term_data.get('original', '')
                    expanded = term_data.get('expanded', {}).get(original, [])
                    
                    display_result += f"**{original}** → "
                    if expanded:
                        expanded_list = [f"{item['word']} ({item['similarity']:.2f})" for item in expanded]
                        display_result += ", ".join(expanded_list)
                        
                        # Add to expanded_words for search
                        if original not in expanded_words:
                            expanded_words[original] = []
                        for item in expanded:
                            expanded_words[original].append(item['word'])
                    else:
                        display_result += "No similar words found."
                    display_result += "\n\n"
        
        # Prepare a JSON structure to be used later in search
        encoded_expanded = json.dumps(expanded_words)
        
        return display_result, encoded_expanded
    except Exception as e:
        logger.error(f"Error expanding boolean expression: {e}")
        return f"Error expanding expression: {str(e)}", ""

# Modify the perform_search_with_keywords function in enhanced_ui.py

def perform_search_with_keywords(
    query: str,
    question: str,
    chunk_size: int,
    year_start: int,
    year_end: int,
    keywords: str,
    search_in: List[str],
    use_semantic_expansion: bool,
    semantic_expansion_factor: int,
    expanded_words_json: str,
    enforce_keywords: bool
) -> Tuple[str, str, str]:
    """Perform search with keyword filtering and semantic expansion."""
    try:
        if not rag_engine:
            return "Error: RAG Engine failed to initialize", "", "Check the logs for details"
        
        start_time = time.time()
        logger.info(f"Starting search with keywords: query='{query}', question='{question}', keywords='{keywords}'")
        
        # Process keywords
        keywords_to_use = keywords.strip() if keywords and keywords.strip() else None
        
        # Use expanded words if provided and semantic expansion is enabled
        expanded_words = None
        if use_semantic_expansion and expanded_words_json:
            try:
                expanded_words = json.loads(expanded_words_json)
                logger.info(f"Using expanded words: {expanded_words}")
            except:
                logger.warning("Failed to parse expanded words JSON")
        
        # Set search fields
        search_fields = search_in if search_in else ["Text"]
        
        # Perform search
        results = rag_engine.search(
            question=question,
            content_description=query,
            year_range=[year_start, year_end],
            chunk_size=chunk_size,
            keywords=keywords_to_use,
            search_in=search_fields,
            use_query_refinement=False,  # Disable for simplicity
            with_citations=False,
            use_semantic_expansion=use_semantic_expansion,
            semantic_expansion_factor=semantic_expansion_factor,
            enforce_keywords=enforce_keywords
        )
        
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.2f} seconds")
        
        # Get actual number of chunks (important for filtered searches)
        num_chunks = len(results.get('chunks', []))
        logger.info(f"Found {num_chunks} chunks")
        
        # Format results
        answer_text = results.get('answer', 'No answer generated')
        
        # Format retrieved chunks for display
        chunks_text = ""
        if results.get("chunks"):
            for i, chunk in enumerate(results["chunks"], 1):
                metadata = chunk["metadata"]
                chunks_text += f"### Chunk {i}\n\n"
                chunks_text += f"**Title**: {metadata.get('Artikeltitel', 'No title')}\n\n"
                chunks_text += f"**Date**: {metadata.get('Datum', 'Unknown')}\n\n"
                
                # Highlight the keywords if found
                if keywords_to_use:
                    content = chunk['content']
                    keywords_list = [k.strip().lower() for k in keywords_to_use.split('AND')]
                    if expanded_words:
                        all_keywords = []
                        for k in keywords_list:
                            all_keywords.append(k)
                            if k in expanded_words:
                                all_keywords.extend(expanded_words[k])
                        
                        # Add a note about which keywords were found
                        found_keywords = []
                        for k in all_keywords:
                            if k.lower() in content.lower():
                                found_keywords.append(k)
                        
                        if found_keywords:
                            chunks_text += f"**Keywords Found**: {', '.join(found_keywords)}\n\n"
                
                chunks_text += f"**Relevance Score**: {chunk['relevance_score']:.4f}\n\n"
                chunks_text += f"**Content**:\n{chunk['content']}\n\n"
                chunks_text += "---\n\n"
        else:
            chunks_text = "No chunks found matching your criteria."
        
        # Format metadata
        metadata_text = f"""
## Search Parameters
- **Query**: {query}
- **Question**: {question}
- **Chunk Size**: {chunk_size}
- **Year Range**: {year_start} - {year_end}
- **Search Time**: {search_time:.2f} seconds
- **Results Found**: {num_chunks}

## Keyword Filtering
- **Keywords**: {keywords_to_use or "None"}
- **Search Fields**: {', '.join(search_fields)}
- **Strict Keyword Filtering**: {"Yes" if enforce_keywords else "No"}
- **Semantic Expansion**: {"Enabled" if use_semantic_expansion else "Disabled"}
"""
        if use_semantic_expansion and expanded_words:
            metadata_text += "\n## Expanded Keywords\n"
            for original, similar in expanded_words.items():
                metadata_text += f"- **{original}** → {', '.join(similar)}\n"
            
        return answer_text, chunks_text, metadata_text
    except Exception as e:
        logger.error(f"Error in search: {e}", exc_info=True)
        return f"Error: {str(e)}", "", "Search failed, check logs for details."

# Create enhanced Gradio interface
with gr.Blocks(title="Enhanced Spiegel RAG") as app:
    gr.Markdown("# Enhanced Spiegel RAG System")
    gr.Markdown("Search the archive with keyword filtering and semantic expansion")
    
    # Hidden state for expanded words
    expanded_words_state = gr.State("")
    
    with gr.Tabs():
        with gr.TabItem("Search"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Basic search inputs
                    query = gr.Textbox(
                        label="Search Query (what to retrieve)",
                        placeholder="Describe what content to find in the archive...",
                        lines=2,
                        value="Berlin Mauer"
                    )
                    
                    question = gr.Textbox(
                        label="Question (what to answer)",
                        placeholder="What do you want to know about the retrieved content?",
                        lines=2,
                        value="Wie wurde die Berliner Mauer in den Medien dargestellt?"
                    )
                    
                    with gr.Row():
                        chunk_size = gr.Dropdown(
                            choices=[300, 600, 1200, 2400, 3000],
                            value=600,
                            label="Chunk Size"
                        )
                    
                    with gr.Row():
                        year_start = gr.Slider(
                            minimum=settings.MIN_YEAR,
                            maximum=settings.MAX_YEAR,
                            value=1960,
                            step=1,
                            label="Start Year"
                        )
                        
                        year_end = gr.Slider(
                            minimum=settings.MIN_YEAR,
                            maximum=settings.MAX_YEAR,
                            value=1965,
                            step=1,
                            label="End Year"
                        )
                    
                    # Keyword filtering section
                    gr.Markdown("### Keyword Filtering")
                    
                    keywords = gr.Textbox(
                        label="Boolean Keyword Expression",
                        placeholder="berlin AND (mauer OR wall) NOT soviet",
                        lines=2,
                        info="Use AND, OR, NOT operators for complex expressions"
                    )
                    
                    with gr.Row():
                        parse_btn = gr.Button("Parse Expression")
                    
                    parsed_output = gr.Markdown(label="Parsed Expression")
                    
                    with gr.Row():
                        search_in = gr.CheckboxGroup(
                            choices=["Text", "Artikeltitel", "Schlagworte"],
                            value=["Text"],
                            label="Search In"
                        )
                    
                    with gr.Row():
                        use_semantic_expansion = gr.Checkbox(
                            label="Use Semantic Expansion",
                            value=True,
                            info="Find and include semantically similar words"
                        )
                        
                        semantic_expansion_factor = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Expansion Factor"
                        )
                    
                    with gr.Row():
                        expand_btn = gr.Button("Find Similar Words")
                    
                    expansion_output = gr.Markdown(label="Similar Words")
                    
                    enforce_keywords = gr.Checkbox(
                        label="Strict Keyword Filtering",
                        value=True,
                        info="Only return results containing the keywords"
                    )
                    
                    search_btn = gr.Button("Search", variant="primary")
                
                with gr.Column(scale=1):
                    # Results
                    answer_output = gr.Markdown(
                        label="Answer",
                        value=""
                    )
                    
                    with gr.Tabs():
                        with gr.TabItem("Retrieved Chunks"):
                            chunks_output = gr.Markdown(
                                value=""
                            )
                        with gr.TabItem("Search Metadata"):
                            metadata_output = gr.Markdown(
                                value=""
                            )
        
        with gr.TabItem("Word Exploration"):
            gr.Markdown("### Explore Similar Words")
            gr.Markdown("Use this tool to find semantically similar words in the corpus")
            
            with gr.Row():
                single_keyword = gr.Textbox(
                    label="Enter a word",
                    placeholder="Type a single word...",
                    lines=1
                )
                
                single_expansion_factor = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Number of similar words"
                )
            
            single_word_btn = gr.Button("Find Similar Words")
            
            single_word_output = gr.Markdown(label="Similar Words")
    
    # Set up event handlers
    parse_btn.click(
        parse_boolean_expression,
        inputs=[keywords],
        outputs=[parsed_output]
    )
    
    expand_btn.click(
        expand_boolean_expression,
        inputs=[keywords, semantic_expansion_factor],
        outputs=[expansion_output, expanded_words_state]
    )
    
    search_btn.click(
        perform_search_with_keywords,
        inputs=[
            query, question, chunk_size, year_start, year_end,
            keywords, search_in, use_semantic_expansion,
            semantic_expansion_factor, expanded_words_state,
            enforce_keywords
        ],
        outputs=[answer_output, chunks_output, metadata_output]
    )
    
    single_word_btn.click(
        find_similar_words,
        inputs=[single_keyword, single_expansion_factor],
        outputs=[single_word_output]
    )

# Run the app
if __name__ == "__main__":
    logger.info("Starting enhanced Gradio UI...")
    app.launch(share=False)