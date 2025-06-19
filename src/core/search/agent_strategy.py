# src/core/search/agent_strategy.py - Updated to preserve both scores
"""
Updated agent search strategy with support for minimum retrieval relevance score
and preservation of both vector similarity and LLM evaluation scores.
"""
import logging
import time
import re
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass

from langchain.docstore.document import Document

from src.core.search.strategies import SearchStrategy, SearchConfig, SearchResult
from src.core.llm_service import LLMService
from src.config import settings

logger = logging.getLogger(__name__)

@dataclass
class AgentSearchConfig:
    """Configuration specific to agent search."""
    use_time_windows: bool = True
    time_window_size: int = 5
    chunks_per_window_initial: int = 50
    chunks_per_window_final: int = 20
    agent_model: str = "hu-llm3"
    agent_system_prompt: str = ""
    evaluation_batch_size: int = 8
    min_retrieval_relevance_score: float = 0.25  # NEW: Minimum retrieval score


class TimeWindowedAgentStrategy(SearchStrategy):
    """
    Agent search strategy with integrated time windowing and dual score preservation.
    
    Now preserves both vector similarity scores and LLM evaluation scores for analysis.
    """
    
    def __init__(self, llm_service: LLMService, agent_config: AgentSearchConfig):
        """
        Initialize the agent strategy.
        
        Args:
            llm_service: Service for LLM operations
            agent_config: Agent-specific configuration
        """
        self.llm_service = llm_service
        self.agent_config = agent_config
        self._cancellation_requested = False
        
    def search(self, 
              config: SearchConfig, 
              vector_store: Any,
              progress_callback: Optional[Callable[[str, float], None]] = None) -> SearchResult:
        """
        Execute time-windowed agent search with dual score preservation.
        
        Args:
            config: Basic search configuration
            vector_store: Vector store interface
            progress_callback: Optional callback for progress updates
            
        Returns:
            SearchResult with evaluated chunks and detailed metadata including both scores
        """
        start_time = time.time()
        self._cancellation_requested = False
        
        try:
            if progress_callback:
                progress_callback("Initialisiere Agenten-Suche...", 0.0)
            
            # Step 1: Determine time windows or use global search
            if self.agent_config.use_time_windows:
                time_windows = self._create_time_windows(config.year_range)
                logger.info(f"Created {len(time_windows)} time windows: {time_windows}")
            else:
                time_windows = [config.year_range]  # Single "window" for entire range
                logger.info("Using global search (no time windows)")
            
            if progress_callback:
                progress_callback(f"Suche in {len(time_windows)} Zeitfenstern...", 0.1)
            
            # Step 2: Retrieve initial chunks per window
            all_initial_chunks = []
            window_chunks_map = {}
            
            for i, window in enumerate(time_windows):
                if self._cancellation_requested:
                    raise InterruptedError("Search cancelled by user")
                
                window_start, window_end = window
                window_key = f"{window_start}-{window_end}"
                
                if progress_callback:
                    progress = 0.1 + (i / len(time_windows)) * 0.3
                    progress_callback(f"Zeitfenster {window_key}: Texte abrufen...", progress)
                
                # Get chunks for this window with UPDATED minimum score
                window_chunks = self._retrieve_chunks_for_window(
                    config, vector_store, window, 
                    self.agent_config.chunks_per_window_initial,
                    self.agent_config.min_retrieval_relevance_score  # NEW: Pass minimum score
                )
                
                logger.info(f"Window {window_key}: Retrieved {len(window_chunks)} initial chunks")
                
                # Add window metadata to chunks
                for doc, score in window_chunks:
                    doc.metadata['time_window'] = window_key
                    doc.metadata['window_start'] = window_start
                    doc.metadata['window_end'] = window_end
                    # IMPORTANT: Store the original vector score for later use
                    doc.metadata['vector_similarity_score'] = score
                
                all_initial_chunks.extend(window_chunks)
                window_chunks_map[window_key] = window_chunks
            
            if not all_initial_chunks:
                logger.warning("No initial chunks retrieved")
                return SearchResult(
                    chunks=[],
                    metadata={
                        "strategy": "time_windowed_agent",
                        "error": "No chunks found in any time window",
                        "search_time": time.time() - start_time,
                        "agent_config": self.agent_config.__dict__
                    }
                )
            
            if progress_callback:
                progress_callback(f"Gesamt: {len(all_initial_chunks)} Texte abgerufen", 0.4)
            
            # Step 3: Evaluate chunks with LLM per window
            final_chunks = []
            all_evaluations = []
            
            for i, (window_key, window_chunks) in enumerate(window_chunks_map.items()):
                if self._cancellation_requested:
                    raise InterruptedError("Search cancelled by user")
                
                if progress_callback:
                    progress = 0.4 + (i / len(window_chunks_map)) * 0.5
                    progress_callback(f"Zeitfenster {window_key}: KI-Bewertung...", progress)
                
                # Evaluate chunks in this window
                evaluated_chunks, window_evaluations = self._evaluate_chunks_in_window(
                    config.content_description, window_chunks, window_key
                )
                
                # Select top chunks for this window - UPDATED to preserve both scores
                top_chunks = self._select_top_chunks_with_dual_scores(
                    evaluated_chunks, self.agent_config.chunks_per_window_final
                )
                
                final_chunks.extend(top_chunks)
                all_evaluations.extend(window_evaluations)
                
                logger.info(f"Window {window_key}: Selected {len(top_chunks)} final chunks")
            
            if progress_callback:
                progress_callback(f"Agenten-Suche abgeschlossen: {len(final_chunks)} Texte ausgewählt", 1.0)
            
            # Step 4: Prepare results with dual score metadata
            search_time = time.time() - start_time
            
            # Create metadata with comprehensive information
            metadata = {
                "strategy": "time_windowed_agent",
                "search_time": search_time,
                "agent_config": self.agent_config.__dict__,
                "time_windows": time_windows,
                "window_chunks_map": {k: len(v) for k, v in window_chunks_map.items()},
                "total_initial_chunks": len(all_initial_chunks),
                "total_final_chunks": len(final_chunks),
                "evaluations": all_evaluations,
                "config": {
                    "year_range": config.year_range,
                    "chunk_size": config.chunk_size,
                    "keywords": config.keywords,
                    "min_retrieval_relevance_score": self.agent_config.min_retrieval_relevance_score  # NEW
                }
            }
            
            logger.info(f"Agent search completed: {len(final_chunks)} chunks selected in {search_time:.2f}s")
            
            return SearchResult(
                chunks=final_chunks,
                metadata=metadata
            )
            
        except InterruptedError:
            logger.info("Agent search was cancelled")
            return SearchResult(
                chunks=[],
                metadata={
                    "strategy": "time_windowed_agent",
                    "error": "Search cancelled by user",
                    "search_time": time.time() - start_time
                }
            )
        except Exception as e:
            logger.error(f"Agent search failed: {e}", exc_info=True)
            return SearchResult(
                chunks=[],
                metadata={
                    "strategy": "time_windowed_agent",
                    "error": str(e),
                    "search_time": time.time() - start_time
                }
            )
    
    def cancel_search(self):
        """Cancel the ongoing search."""
        self._cancellation_requested = True
        logger.info("Agent search cancellation requested")
    
    def _create_time_windows(self, year_range: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Create time windows based on configuration."""
        start_year, end_year = year_range
        window_size = self.agent_config.time_window_size
        
        windows = []
        for window_start in range(start_year, end_year + 1, window_size):
            window_end = min(window_start + window_size - 1, end_year)
            windows.append((window_start, window_end))
        
        return windows
    
    def _retrieve_chunks_for_window(self, 
                                   config: SearchConfig, 
                                   vector_store: Any, 
                                   window: Tuple[int, int], 
                                   chunk_count: int,
                                   min_relevance_score: float = 0.25) -> List[Tuple[Document, float]]:
        """Retrieve chunks for a specific time window with minimum relevance score."""
        window_start, window_end = window
        
        # Build filter for this window
        filter_dict = vector_store.build_metadata_filter(
            year_range=[window_start, window_end],
            keywords=None,
            search_in=None
        )
        
        # Perform search for this window with UPDATED minimum score
        chunks = vector_store.similarity_search(
            query=config.content_description,
            chunk_size=config.chunk_size,
            k=chunk_count,
            filter_dict=filter_dict,
            min_relevance_score=min_relevance_score,  # NEW: Use configurable minimum score
            keywords=config.keywords,
            search_in=config.search_fields,
            enforce_keywords=config.enforce_keywords
        )
        
        return chunks
    
    def _evaluate_chunks_in_window(self, 
                                  question: str, 
                                  chunks: List[Tuple[Document, float]], 
                                  window_key: str) -> Tuple[List[Tuple[Document, float, float, str]], List[Dict]]:
        """
        Evaluate chunks using LLM with FULL content and adaptive batching.
        """
        if not chunks:
            return [], []
        
        evaluated_chunks = []
        evaluations = []
        
        # Create adaptive batches based on actual content length
        chunk_batches = self._create_adaptive_batches(chunks)
        
        logger.info(f"Evaluating {len(chunks)} chunks in {len(chunk_batches)} adaptive batches for window {window_key}")
        
        for batch_idx, batch in enumerate(chunk_batches):
            if self._cancellation_requested:
                break
                
            logger.info(f"Processing batch {batch_idx + 1}/{len(chunk_batches)} "
                       f"({len(batch)} chunks) in window {window_key}")
            
            try:
                batch_results = self._evaluate_single_batch(question, batch, window_key)
                evaluated_chunks.extend([r[0] for r in batch_results])
                evaluations.extend([r[1] for r in batch_results])
                
                # Small delay to avoid overwhelming the LLM service
                time.sleep(0.3)
                
            except Exception as e:
                logger.error(f"Batch evaluation failed in window {window_key}: {e}")
                # Add fallback results
                fallback_results = self._create_fallback_results(batch, window_key)
                evaluated_chunks.extend([r[0] for r in fallback_results])
                evaluations.extend([r[1] for r in fallback_results])
        
        return evaluated_chunks, evaluations
    
    def _create_adaptive_batches(self, 
                                chunks: List[Tuple[Document, float]], 
                                max_tokens_per_batch: int = 12000) -> List[List[Tuple[Document, float]]]:
        """Create adaptive batches based on actual content length."""
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        # Reserve tokens for prompt structure and response
        base_prompt_tokens = 1000
        response_tokens = 2000
        available_tokens = max_tokens_per_batch - base_prompt_tokens - response_tokens
        
        for chunk_tuple in chunks:
            doc, score = chunk_tuple
            chunk_tokens = self._estimate_token_count(doc.page_content)
            
            # Check if adding this chunk would exceed limit
            if current_tokens + chunk_tokens > available_tokens and current_batch:
                # Start new batch
                batches.append(current_batch)
                current_batch = [chunk_tuple]
                current_tokens = chunk_tokens
            else:
                # Add to current batch
                current_batch.append(chunk_tuple)
                current_tokens += chunk_tokens
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} adaptive batches from {len(chunks)} chunks")
        for i, batch in enumerate(batches):
            total_chars = sum(len(doc.page_content) for doc, _ in batch)
            logger.debug(f"Batch {i+1}: {len(batch)} chunks, ~{total_chars:,} characters")
        
        return batches
    
    def _estimate_token_count(self, text: str) -> int:
        """Rough estimation of token count for text."""
        # Simple estimation: 1 token ≈ 4 characters
        # This is conservative - actual tokenization varies by model
        return len(text) // 4
    
    def _evaluate_single_batch(self, 
                              question: str, 
                              batch: List[Tuple[Document, float]], 
                              window_key: str) -> List[Tuple[Tuple[Document, float, float, str], Dict]]:
        """Evaluate a single batch of chunks with FULL content."""
        
        # Calculate optimal batch size based on chunk sizes and token limits
        optimal_batch_size = self._calculate_optimal_batch_size(batch)
        
        if optimal_batch_size < len(batch):
            # Split batch if it's too large for token limits
            logger.info(f"Splitting batch of {len(batch)} into smaller batches of {optimal_batch_size}")
            sub_batches = [batch[i:i + optimal_batch_size] for i in range(0, len(batch), optimal_batch_size)]
            
            all_results = []
            for sub_batch in sub_batches:
                sub_results = self._evaluate_single_batch(question, sub_batch, window_key)
                all_results.extend(sub_results)
            return all_results
        
        # Prepare enhanced system prompt
        system_prompt = self._create_enhanced_system_prompt()
        
        # Create comprehensive evaluation prompt with FULL chunks
        batch_content = self._create_full_content_prompt(question, batch, window_key)
        
        # Get LLM evaluation
        try:
            response = self.llm_service.generate_response(
                question=batch_content,
                context="",
                model=self.agent_config.agent_model,
                system_prompt=system_prompt,
                temperature=0.2
                )
            
            response_text = response.get('text', '')
            logger.info(f"LLM Response for batch in {window_key}: {response_text[:200]}...")
            
            # Parse response and create results
            return self._parse_batch_evaluation(batch, response_text, window_key)
            
        except Exception as e:
            logger.error(f"LLM evaluation failed for batch in {window_key}: {e}")
            # Return fallback scores
            return self._create_fallback_results(batch, window_key)
    
    def _calculate_optimal_batch_size(self, batch: List[Tuple[Document, float]]) -> int:
        """Calculate optimal batch size based on chunk lengths and token limits."""
        
        # Estimate tokens per chunk (rough estimate: 1 token ≈ 4 characters)
        avg_chunk_length = sum(len(doc.page_content) for doc, _ in batch) / len(batch)
        estimated_tokens_per_chunk = avg_chunk_length / 4
        
        # Conservative token limits (leave room for prompt, response, etc.)
        max_input_tokens = 16000  # Conservative limit for most models
        tokens_for_prompt_and_response = 4000
        available_tokens = max_input_tokens - tokens_for_prompt_and_response
        
        # Calculate how many chunks we can fit
        optimal_size = min(
            len(batch),
            max(1, int(available_tokens / estimated_tokens_per_chunk))
        )
        
        logger.debug(f"Avg chunk length: {avg_chunk_length:.0f} chars, "
                    f"Est. tokens/chunk: {estimated_tokens_per_chunk:.0f}, "
                    f"Optimal batch size: {optimal_size}")
        
        return optimal_size
    
    def _create_enhanced_system_prompt(self) -> str:
        """Create an enhanced system prompt for full chunk evaluation."""
        
        if self.agent_config.agent_system_prompt:
            return self.agent_config.agent_system_prompt
        
        return """Du bist ein Experte für historische Forschung und Medienanalyse, spezialisiert auf deutsche Nachkriegsgeschichte.

AUFGABE: Bewerte vollständige Textabschnitte nach ihrer Relevanz für eine Forschungsfrage.

BEWERTUNGSKRITERIEN:
- 9-10/10: Direkt relevante Texte (explizite und ausführliche Behandlung des Themas)
- 7-8/10: Stark relevante Texte (thematisch verwandt, wichtiger Kontext, teilweise Behandlung)
- 5-6/10: Mäßig relevante Texte (indirekter Bezug, relevanter historischer Kontext)
- 3-4/10: Schwach relevante Texte (entfernter Bezug, allgemeiner zeithistorischer Hintergrund)
- 1-2/10: Nicht relevante Texte (kein erkennbarer thematischer Bezug)

WICHTIGE HINWEISE:
- Du erhältst die VOLLSTÄNDIGEN Textabschnitte zur Bewertung
- Lies jeden Text sorgfältig und vollständig durch
- Achte auf relevante Details auch am Ende des Textes
- Berücksichtige Kontext, Schlüsselwörter und thematische Verbindungen
- Auch indirekte Bezüge und zeithistorischer Kontext können wertvoll sein
- Bewerte großzügig aber differenziert

ANTWORTFORMAT (EXAKT EINHALTEN):
Text 1: Score X/10 - Kurze präzise Begründung
Text 2: Score Y/10 - Kurze präzise Begründung
...

Verwende Scores von 1-10. Sei gründlich aber präzise in der Bewertung."""
    
    def _create_full_content_prompt(self, 
                                   question: str, 
                                   batch: List[Tuple[Document, float]], 
                                   window_key: str) -> str:
        """Create evaluation prompt with FULL chunk content."""
        
        prompt = f"""FORSCHUNGSFRAGE: {question}
ZEITFENSTER: {window_key}

VOLLSTÄNDIGE TEXTABSCHNITTE ZUR BEWERTUNG:
{'='*60}

"""
        
        for i, (doc, score) in enumerate(batch):
            title = doc.metadata.get('Artikeltitel', 'Kein Titel')
            date = doc.metadata.get('Datum', 'Unbekannt')
            
            # Use FULL content - this is the key fix!
            full_content = doc.page_content.strip()
            
            prompt += f"""Text {i+1}: {title} ({date})
{'-'*40}
{full_content}

{'='*60}

"""
        
        prompt += f"""BEWERTE alle {len(batch)} Texte vollständig und gründlich.

Format: Text X: Score Y/10 - Begründung

Berücksichtige den gesamten Textinhalt für deine Bewertung."""
        
        return prompt
    
    def _create_fallback_results(self, 
                                batch: List[Tuple[Document, float]], 
                                window_key: str) -> List[Tuple[Tuple[Document, float, float, str], Dict]]:
        """Create fallback results when LLM evaluation fails."""
        
        results = []
        for i, (doc, vector_score) in enumerate(batch):
            # Use vector score as fallback
            fallback_score = min(0.8, max(0.3, vector_score))
            eval_text = "Score: 5/10 - Fallback-Bewertung (LLM-Evaluierung fehlgeschlagen)"
            
            chunk_result = (doc, vector_score, fallback_score, eval_text)
            
            evaluation_metadata = {
                "title": doc.metadata.get('Artikeltitel', 'Unknown'),
                "date": doc.metadata.get('Datum', 'Unknown'),
                "window": window_key,
                "vector_score": vector_score,
                "llm_score": fallback_score,
                "original_llm_score": 5.0,
                "evaluation": "Fallback-Bewertung",
                "parsing_success": False,
                "error": "LLM evaluation failed"
            }
            
            results.append((chunk_result, evaluation_metadata))
        
        return results
    
    def _parse_batch_evaluation(self, 
                               batch: List[Tuple[Document, float]], 
                               response_text: str, 
                               window_key: str) -> List[Tuple[Tuple[Document, float, float, str], Dict]]:
        """Parse LLM evaluation response into structured results with robust parsing."""
        results = []
        
        # Multiple regex patterns to catch different response formats
        patterns = [
            r'Text (\d+):\s*Score\s*(\d+(?:\.\d+)?)/10\s*[-–]\s*(.+?)(?=Text \d+:|$)',
            r'Text (\d+):\s*Score\s*(\d+(?:\.\d+)?)\s*[-–]\s*(.+?)(?=Text \d+:|$)',
            r'Text (\d+):\s*(\d+(?:\.\d+)?)/10\s*[-–]\s*(.+?)(?=Text \d+:|$)',
            r'Text (\d+):\s*(\d+(?:\.\d+)?)\s*[-–]\s*(.+?)(?=Text \d+:|$)',
            r'(\d+)\.\s*(\d+(?:\.\d+)?)/10\s*[-–]\s*(.+?)(?=\d+\.|$)',
            r'(\d+)\.\s*(\d+(?:\.\d+)?)\s*[-–]\s*(.+?)(?=\d+\.|$)'
        ]
        
        evaluations = {}
        
        # Try each pattern until we find matches
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if matches:
                logger.info(f"Found {len(matches)} evaluations using pattern: {pattern}")
                for match in matches:
                    try:
                        text_num = int(match[0])
                        score = float(match[1])
                        explanation = match[2].strip()
                        evaluations[text_num] = (score, explanation)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse match {match}: {e}")
                break
        
        # If no pattern worked, try to extract any numbers and use them as fallback
        if not evaluations:
            logger.warning(f"No evaluations found with patterns, trying fallback parsing for window {window_key}")
            # Look for any score-like patterns
            numbers = re.findall(r'(\d+(?:\.\d+)?)', response_text)
            scores = [float(n) for n in numbers if 1 <= float(n) <= 10]
            
            # Assign scores in order if we have any
            for i, score in enumerate(scores[:len(batch)]):
                evaluations[i + 1] = (score, "Automatisch extrahiert")
        
        # Log what we found
        logger.info(f"Window {window_key}: Extracted evaluations for texts: {list(evaluations.keys())}")
        
        # Process each chunk in batch
        for i, (doc, vector_score) in enumerate(batch):
            text_num = i + 1
            
            if text_num in evaluations:
                eval_score_raw, explanation = evaluations[text_num]
                # Normalize to 0-1
                eval_score = max(0.0, min(1.0, eval_score_raw / 10.0))
                eval_text = f"Score: {eval_score_raw}/10 - {explanation}"
                logger.debug(f"Text {text_num}: Score {eval_score_raw}/10 -> {eval_score}")
            else:
                # Fallback if not found - use vector score as basis
                eval_score = min(0.8, max(0.3, vector_score))  # Keep reasonable bounds
                eval_text = "Score: 5/10 - Standard-Bewertung (nicht in Antwort gefunden)"
                logger.warning(f"No evaluation found for text {text_num} in window {window_key}, using fallback")
            
            # Create chunk tuple
            chunk_result = (doc, vector_score, eval_score, eval_text)
            
            # Create evaluation metadata
            evaluation_metadata = {
                "title": doc.metadata.get('Artikeltitel', 'Unknown'),
                "date": doc.metadata.get('Datum', 'Unknown'),
                "window": window_key,
                "vector_score": vector_score,
                "llm_score": eval_score,
                "original_llm_score": eval_score_raw if text_num in evaluations else 5.0,
                "evaluation": explanation if text_num in evaluations else "Standard-Bewertung",
                "parsing_success": text_num in evaluations
            }
            
            results.append((chunk_result, evaluation_metadata))
        
        return results
    
    def _select_top_chunks_with_dual_scores(self, 
                                           evaluated_chunks: List[Tuple[Document, float, float, str]], 
                                           target_count: int) -> List[Tuple[Document, float]]:
        """Select top chunks based on LLM scores while preserving both score types."""
        # Sort by LLM score (third element in tuple)
        sorted_chunks = sorted(evaluated_chunks, key=lambda x: x[2], reverse=True)
        
        # Take top chunks and convert to standard format while preserving both scores
        top_chunks = []
        for doc, vector_score, llm_score, eval_text in sorted_chunks[:target_count]:
            # Store BOTH scores in metadata for download and analysis
            doc.metadata['llm_evaluation_score'] = llm_score
            doc.metadata['evaluation_text'] = eval_text
            # KEEP the original vector score that was already stored during retrieval
            # doc.metadata['vector_similarity_score'] should already be there from _retrieve_chunks_for_window
            
            # Use LLM score as the primary relevance score for UI display
            top_chunks.append((doc, llm_score))
        
        return top_chunks