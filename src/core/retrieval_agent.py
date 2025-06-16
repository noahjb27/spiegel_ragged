# src/core/retrieval_agent.py - Fixed version
"""
Simplified RetrievalAgent that works reliably with HU-LLM.
No complex JSON schemas - just simple, reliable scoring.
"""
import logging
import time
import re
from typing import Dict, List, Optional, Tuple, Any

from langchain.docstore.document import Document

from src.core.vector_store import ChromaDBInterface
from src.core.llm_service import LLMService
from src.config import settings

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Simplified RetrievalAgent that works reliably with HU-LLM.
    Uses simple scoring instead of complex JSON schemas.
    """
    
    def __init__(self, vector_store: ChromaDBInterface, llm_service: LLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service
        logger.info("Initialized Simplified RetrievalAgent - optimized for HU-LLM")
    
    def retrieve_and_refine(
        self,
        question: str,
        content_description: str,
        year_range: Optional[List[int]] = None,
        chunk_size: Optional[int] = None,
        keywords: Optional[str] = None,
        search_in: Optional[List[str]] = None,
        enforce_keywords: bool = True,
        initial_retrieval_count: int = 100,
        filter_stages: List[int] = [50, 20, 10],
        model: str = "hu-llm3",  # FIXED: Updated default model
        with_evaluations: bool = True
    ) -> Tuple[List[Tuple[Document, float, float, str]], Dict[str, Any]]:
        """
        Simplified retrieval and refinement that works reliably with HU-LLM.
        FIXED: Removed openai_api_key parameter
        """
        start_time = time.time()
        stage_times = []
        stage_results = []
        
        # Use defaults if not provided
        chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        year_range = year_range or [settings.MIN_YEAR, settings.MAX_YEAR]
        
        logger.info(f"Starting simplified retrieval agent with question: '{question}'")
        logger.info(f"Initial retrieval count: {initial_retrieval_count}, Filter stages: {filter_stages}")
        
        # Step 1: Initial retrieval
        retrieval_start = time.time()
        filter_dict = self.vector_store.build_metadata_filter(
            year_range=year_range,
            keywords=None,
            search_in=None
        )
        
        initial_chunks = self.vector_store.similarity_search(
            query=content_description,
            chunk_size=chunk_size,
            k=initial_retrieval_count,
            filter_dict=filter_dict,
            min_relevance_score=0.25,
            keywords=keywords,
            search_in=search_in,
            enforce_keywords=enforce_keywords
        )
        
        retrieval_time = time.time() - retrieval_start
        stage_times.append(("Initial Retrieval", retrieval_time))
        stage_results.append(len(initial_chunks))
        
        logger.info(f"Initial retrieval: {len(initial_chunks)} chunks in {retrieval_time:.2f}s")
        
        if not initial_chunks:
            logger.warning("No initial chunks found, returning empty results")
            return [], {
                "success": False,
                "error": "No relevant content found for initial retrieval",
                "total_time": time.time() - start_time,
                "stage_times": stage_times,
                "stage_results": stage_results
            }
        
        current_chunks = initial_chunks
        
        # Step 2: Simplified iterative refinement
        for i, target_count in enumerate(filter_stages):
            stage_start = time.time()
            
            if len(current_chunks) <= target_count:
                logger.info(f"Skipping filter stage {i+1} as we already have {len(current_chunks)} ≤ {target_count} chunks")
                continue
                
            logger.info(f"Starting simplified filter stage {i+1}: {len(current_chunks)} → {target_count} chunks")
            
            # Use simplified batch evaluation - FIXED: Removed openai_api_key parameter
            evaluated_chunks = self._evaluate_chunks_simple_batch(
                question=question,
                chunks=current_chunks,
                model=model
            )
            
            # Sort by evaluation score and keep top chunks
            evaluated_chunks.sort(key=lambda x: x[2], reverse=True)
            current_chunks = [(doc, score, eval_text) for doc, score, eval_score, eval_text in evaluated_chunks[:target_count]]
            
            stage_time = time.time() - stage_start
            stage_times.append((f"Simplified Filter Stage {i+1}", stage_time))
            stage_results.append(len(current_chunks))
            
            logger.info(f"Completed simplified filter stage {i+1}: reduced to {len(current_chunks)} chunks in {stage_time:.2f}s")
        
        # Return final results
        total_time = time.time() - start_time
        logger.info(f"Simplified retrieval agent completed in {total_time:.2f}s")
        
        # Convert to expected format
        final_chunks = []
        for doc, vector_score, eval_text in current_chunks:
            # Extract score from eval_text if available
            score_match = re.search(r'Score: ([\d.]+)', eval_text)
            eval_score = float(score_match.group(1)) / 10.0 if score_match else 0.5
            final_chunks.append((doc, vector_score, eval_score, eval_text))
        
        metadata = {
            "success": True,
            "total_time": total_time,
            "stage_times": stage_times,
            "stage_results": stage_results,
            "initial_retrieval_count": initial_retrieval_count,
            "filter_stages": filter_stages,
            "final_chunk_count": len(final_chunks),
            "approach": "simplified_batch_evaluation"
        }
        
        return final_chunks, metadata
    
    def _evaluate_chunks_simple_batch(
        self,
        question: str,
        chunks: List,
        model: str = "hu-llm3",  # FIXED: Updated default model
        batch_size: int = 10  # Smaller batches
    ) -> List[Tuple[Document, float, float, str]]:
        """
        Simplified batch evaluation that works reliably with HU-LLM.
        Uses simple text-based scoring instead of JSON schemas.
        FIXED: Removed openai_api_key parameter
        """
        if not chunks:
            return []
            
        all_evaluated_chunks = []
        
        # Process chunks in batches
        chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        logger.info(f"Evaluating {len(chunks)} chunks in {len(chunk_batches)} batches of max {batch_size}")
        
        for batch_idx, batch in enumerate(chunk_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(chunk_batches)} with {len(batch)} chunks")
            
            try:
                # FIXED: Removed openai_api_key parameter
                batch_result = self._evaluate_single_batch(question, batch, model)
                all_evaluated_chunks.extend(batch_result)
                
                # Add small delay to avoid overwhelming the service
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} evaluation failed: {e}")
                # Add fallback scores for failed batch
                for chunk_data in batch:
                    if len(chunk_data) == 2:
                        doc, vector_score = chunk_data
                    else:
                        doc, vector_score, _ = chunk_data
                    
                    all_evaluated_chunks.append((doc, vector_score, 0.5, "Evaluation failed, using default score"))
        
        logger.info(f"Completed evaluation of {len(all_evaluated_chunks)} chunks")
        return all_evaluated_chunks
    
    def _evaluate_single_batch(
        self,
        question: str,
        batch: List,
        model: str
    ) -> List[Tuple[Document, float, float, str]]:
        """
        Evaluate a single batch using simple text-based approach.
        FIXED: Removed openai_api_key parameter
        """
        # Create a simple evaluation prompt
        system_prompt = """Du bist ein Experte für die Bewertung historischer Dokumente. 
Bewerte jeden Textabschnitt für seine Relevanz zur Forschungsfrage auf einer Skala von 0-10.

Antworte für jeden Text in diesem Format:
Text 1: Score X - Kurze Begründung
Text 2: Score Y - Kurze Begründung
etc.

Sei kritisch - die meisten Texte sollten 3-7 bekommen, nur außergewöhnliche erhalten 8-10."""
        
        # Prepare batch content
        batch_content = f"Forschungsfrage: {question}\n\n"
        batch_content += "Bewerte diese Textabschnitte:\n\n"
        
        for i, chunk_data in enumerate(batch):
            if len(chunk_data) == 2:
                doc, score = chunk_data
            else:
                doc, score, _ = chunk_data
            
            title = doc.metadata.get('Artikeltitel', 'Kein Titel')[:80]
            date = doc.metadata.get('Datum', 'Unbekannt')
            content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            
            batch_content += f"Text {i+1}: {title} ({date})\n{content_preview}\n\n"
        
        batch_content += f"Bewerte alle {len(batch)} Texte in dem angegebenen Format."
        
        try:
            # Use LLM service - FIXED: Removed openai_api_key parameter
            response = self.llm_service.generate_response(
                question=batch_content,
                context="",
                model=model,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=10000  # Enough for simple responses
            )
            
            response_text = response.get('text', '')
            
            # Parse the simple text response
            return self._parse_simple_response(batch, response_text)
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            raise
    
    def _parse_simple_response(
        self, 
        batch: List, 
        response_text: str
    ) -> List[Tuple[Document, float, float, str]]:
        """
        Parse simple text response into scores.
        """
        results = []
        
        # Extract scores using regex
        score_pattern = r'Text (\d+):\s*Score\s*(\d+(?:\.\d+)?)\s*-\s*(.+?)(?=Text \d+:|$)'
        matches = re.findall(score_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        # Create a mapping of text number to evaluation
        evaluations = {}
        for match in matches:
            text_num = int(match[0])
            score = float(match[1])
            explanation = match[2].strip()
            evaluations[text_num] = (score, explanation)
        
        # Process each chunk
        for i, chunk_data in enumerate(batch):
            if len(chunk_data) == 2:
                doc, vector_score = chunk_data
            else:
                doc, vector_score, _ = chunk_data
            
            text_num = i + 1
            
            if text_num in evaluations:
                eval_score_raw, explanation = evaluations[text_num]
                # Normalize to 0-1
                eval_score = max(0.0, min(1.0, eval_score_raw / 10.0))
                eval_text = f"Score: {eval_score_raw}/10 - {explanation}"
            else:
                # Fallback if not found in response
                eval_score = 0.5
                eval_text = f"Score: 5/10 - Default score (evaluation not found in response)"
                logger.warning(f"No evaluation found for text {text_num}")
            
            results.append((doc, vector_score, eval_score, eval_text))
        
        logger.info(f"Parsed {len(results)} evaluations from response")
        return results