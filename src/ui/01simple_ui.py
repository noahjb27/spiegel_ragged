# src/ui/simple_ui.py
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
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize RAG engine
try:
    logger.info("Initializing RAG Engine...")
    rag_engine = SpiegelRAGEngine()
    logger.info("RAG Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG Engine: {e}")
    rag_engine = None

def perform_simple_search(
    query: str,
    question: str,
    chunk_size: int,
    year_start: int,
    year_end: int,
) -> Tuple[str, str]:
    """Simplified search function for testing core functionality."""
    try:
        if not rag_engine:
            return "Error: RAG Engine failed to initialize", "Check the logs for details"
        
        start_time = time.time()
        logger.info(f"Starting simple search: query='{query}', question='{question}'")
        
        # Perform search with minimal options
        results = rag_engine.search(
            question=question,
            content_description=query,
            year_range=[year_start, year_end],
            chunk_size=chunk_size,
            use_query_refinement=False,  # Disable for faster results
            with_citations=False,
            use_semantic_expansion=False  # Disable for faster results
        )
        
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.2f} seconds")
        logger.info(f"Found {len(results.get('chunks', []))} chunks")
        
        # Format results
        chunks_text = ""
        if results.get("chunks"):
            for i, chunk in enumerate(results["chunks"], 1):
                metadata = chunk["metadata"]
                chunks_text += f"--- Chunk {i} ---\n"
                chunks_text += f"Title: {metadata.get('Artikeltitel', 'No title')}\n"
                chunks_text += f"Date: {metadata.get('Datum', 'Unknown')}\n"
                chunks_text += f"Score: {chunk['relevance_score']:.4f}\n"
                chunks_text += f"Content: {chunk['content'][:200]}...\n\n"
        else:
            chunks_text = "No chunks found."
        
        summary = f"""
Search Summary:
--------------
Query: {query}
Question: {question}
Time taken: {search_time:.2f} seconds
Results found: {len(results.get('chunks', []))}
Answer length: {len(results.get('answer', ''))} characters
--------------
Answer:

{results.get('answer', 'No answer generated')}
"""
        
        return summary, chunks_text
    except Exception as e:
        logger.error(f"Error in simple search: {e}", exc_info=True)
        return f"Error: {str(e)}", "Search failed, check logs for details."

# Create simplified Gradio interface
with gr.Blocks(title="Simple Spiegel RAG Tester") as app:
    gr.Markdown("# Spiegel RAG System Tester")
    gr.Markdown("A simplified interface to test core RAG functionality")
    
    with gr.Row():
        with gr.Column():
            query = gr.Textbox(
                label="Search Query (what to retrieve)",
                placeholder="Describe what content to find in the archive...",
                lines=2,
                value="Berlin Mauer"  # Default value for testing
            )
            
            question = gr.Textbox(
                label="Question (what to answer)",
                placeholder="What do you want to know about the retrieved content?",
                lines=2,
                value="Wie wurde die Berliner Mauer in den Medien dargestellt?"  # Default value
            )
            
            with gr.Row():
                chunk_size = gr.Dropdown(
                    choices=[300, 600, 1200, 2400, 3000],
                    value=600,  # Smaller size for faster testing
                    label="Chunk Size"
                )
            
            with gr.Row():
                year_start = gr.Slider(
                    minimum=settings.MIN_YEAR,
                    maximum=settings.MAX_YEAR,
                    value=1960,  # Narrower range for testing
                    step=1,
                    label="Start Year"
                )
                
                year_end = gr.Slider(
                    minimum=settings.MIN_YEAR,
                    maximum=settings.MAX_YEAR,
                    value=1965,  # Narrower range for testing
                    step=1,
                    label="End Year"
                )
            
            search_btn = gr.Button("Search", variant="primary")
        
        with gr.Column():
            answer_output = gr.Textbox(
                label="Search Results",
                lines=20,
                max_lines=50
            )
            
            chunks_output = gr.Textbox(
                label="Retrieved Chunks",
                lines=20,
                max_lines=50
            )
    
    # Set up event handler
    search_btn.click(
        perform_simple_search,
        inputs=[query, question, chunk_size, year_start, year_end],
        outputs=[answer_output, chunks_output]
    )

# Run the app
if __name__ == "__main__":
    logger.info("Starting simplified Gradio UI...")
    app.launch(share=False)