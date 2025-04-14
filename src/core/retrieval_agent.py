"""
Retrieval Agent for iterative, LLM-guided document retrieval and refinement.
Implements a hybrid RAG approach with multi-stage filtering of retrieved chunks.
"""
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union

from langchain.docstore.document import Document
from src.core.vector_store import ChromaDBInterface
from src.core.llm_service import LLMService
from src.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class RetrievalAgent:
    """Agent that combines vector retrieval with LLM-guided filtering and reranking."""
    
    def __init__(self, vector_store: ChromaDBInterface, llm_service: LLMService):
        """
        Initialize retrieval agent with vector store and LLM services.
        
        Args:
            vector_store: ChromaDBInterface instance
            llm_service: LLMService instance
        """
        self.vector_store = vector_store
        self.llm_service = llm_service
        logger.info("Initialized RetrievalAgent")
    
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
        model: str = "hu-llm",
        openai_api_key: Optional[str] = None,
        with_evaluations: bool = True
    ) -> Tuple[List[Tuple[Document, float, Optional[str]]], Dict[str, Any]]:
        """
        Perform staged retrieval and refinement of chunks.
        
        Args:
            question: User's question
            content_description: Description for initial retrieval
            year_range: Range of years to search
            chunk_size: Size of chunks to retrieve
            keywords: Optional boolean expression for keyword filtering
            search_in: Where to search for keywords
            enforce_keywords: Whether to enforce keyword presence
            initial_retrieval_count: Number of chunks to retrieve initially
            filter_stages: List of chunk counts for each filtering stage
            model: LLM model to use for evaluation
            openai_api_key: OpenAI API key if using OpenAI models
            with_evaluations: Whether to include evaluation explanations
            
        Returns:
            Tuple of (refined_chunks, metadata)
        """
        start_time = time.time()
        stage_times = []
        stage_results = []
        
        # Use defaults if not provided
        chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        year_range = year_range or [settings.MIN_YEAR, settings.MAX_YEAR]
        
        logger.info(f"Starting retrieval agent with question: '{question}'")
        logger.info(f"Initial retrieval count: {initial_retrieval_count}, Filter stages: {filter_stages}")
        
        # Step 1: Initial retrieval of a larger set of chunks
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
            min_relevance_score=0.25,  # Lower threshold for initial retrieval
            keywords=keywords,
            search_in=search_in,
            enforce_keywords=enforce_keywords
        )
        
        retrieval_time = time.time() - retrieval_start
        stage_times.append(("Initial Retrieval", retrieval_time))
        stage_results.append(len(initial_chunks))
        
        logger.info(f"Initial retrieval: {len(initial_chunks)} chunks in {retrieval_time:.2f}s")
        
        # Check if we got enough initial results
        if not initial_chunks:
            logger.warning("No initial chunks found, returning empty results")
            return [], {
                "success": False,
                "error": "No relevant content found for initial retrieval",
                "total_time": time.time() - start_time,
                "stage_times": stage_times,
                "stage_results": stage_results
            }
        
        # Convert chunks to format for evaluation
        current_chunks = initial_chunks
        
        # Step 2: Iterative refinement through each filter stage
        for i, target_count in enumerate(filter_stages):
            stage_start = time.time()
            
            # Skip if we already have fewer chunks than the target
            if len(current_chunks) <= target_count:
                logger.info(f"Skipping filter stage {i+1} as we already have {len(current_chunks)} ≤ {target_count} chunks")
                continue
                
            logger.info(f"Starting filter stage {i+1}: {len(current_chunks)} → {target_count} chunks")
            
            # Evaluate and rank chunks for this stage
            evaluated_chunks = self._evaluate_chunks(
            question=question,
            chunks=current_chunks,
            model=model,
            openai_api_key=openai_api_key,
            batch_size=min(25, len(current_chunks))  # Process in batches
        )

            # Sort by evaluation score and keep top chunks
            evaluated_chunks.sort(key=lambda x: x[2], reverse=True)
            current_chunks = [(doc, score, eval_text) for doc, score, eval_score, eval_text in evaluated_chunks[:target_count]]
            
            stage_time = time.time() - stage_start
            stage_times.append((f"Filter Stage {i+1}", stage_time))
            stage_results.append(len(current_chunks))
            
            logger.info(f"Completed filter stage {i+1}: reduced to {len(current_chunks)} chunks in {stage_time:.2f}s")
        
        # Return final results
        total_time = time.time() - start_time
        logger.info(f"Retrieval agent completed in {total_time:.2f}s")
        
        # If not including evaluations, remove them from results
        if not with_evaluations:
            final_chunks = [(doc, score, None) for doc, score, eval_text in current_chunks]
        else:
            final_chunks = current_chunks
        
        metadata = {
            "success": True,
            "total_time": total_time,
            "stage_times": stage_times,
            "stage_results": stage_results,
            "initial_retrieval_count": initial_retrieval_count,
            "filter_stages": filter_stages,
            "final_chunk_count": len(final_chunks)
        }
        
        return final_chunks, metadata
    
    def _evaluate_chunks(
        self,
        question: str,
        chunks: List,  # Use generic List since format varies
        model: str = "hu-llm",
        openai_api_key: Optional[str] = None,
        batch_size: int = 25
    ) -> List[Tuple[Document, float, float, str]]:
        """
        Evaluate chunks using LLM to determine relevance to the question.
        
        Args:
            question: User's question
            chunks: List of chunk tuples (format varies)
            model: LLM model to use
            openai_api_key: OpenAI API key if using OpenAI models
            batch_size: Number of chunks to evaluate in each batch
            
        Returns:
            List of (Document, vector_score, eval_score, eval_text) tuples
        """
        if not chunks:
            return []
            
        # Process chunks in batches
        all_evaluated_chunks = []
        
        # Split into batches
        chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        logger.info(f"Evaluating chunks in {len(chunk_batches)} batches of up to {batch_size}")
        
        for batch_idx, batch in enumerate(chunk_batches):
            logger.info(f"Evaluating batch {batch_idx + 1}/{len(chunk_batches)}")
            batch_result = self._evaluate_chunk_batch(question, batch, model, openai_api_key)
            all_evaluated_chunks.extend(batch_result)
            
        return all_evaluated_chunks      
     
    def _evaluate_chunk_batch(
        self,
        question: str,
        chunks: List,  # Use a generic List type since format varies
        model: str,
        openai_api_key: Optional[str] = None
    ) -> List[Tuple[Document, float, float, str]]:
        """
        Evaluate a batch of chunks using LLM.
        
        Args:
            question: User's question
            chunks: List of chunk tuples (either (doc, score) or (doc, score, eval_text))
            model: LLM model to use
            openai_api_key: OpenAI API key if using OpenAI models
            
        Returns:
            List of (Document, vector_score, eval_score, eval_text) tuples
        """
        # Construct a prompt for batch evaluation
        chunk_texts = []
        
        for i, chunk_data in enumerate(chunks):
            # Handle different chunk formats - either (doc, score) or (doc, score, eval_text)
            if len(chunk_data) == 2:
                doc, score = chunk_data
                eval_text = None
            else:
                doc, score, eval_text = chunk_data
            
            metadata = doc.metadata
            chunk_text = f"[CHUNK {i+1}]\nTitle: {metadata.get('Artikeltitel', 'No title')}\n"
            chunk_text += f"Date: {metadata.get('Datum', 'Unknown')}\n"
            chunk_text += f"Content: {doc.page_content}\n"
            chunk_texts.append(chunk_text)
        
        # Rest of the method remains the same...
        prompt = f"""Evaluate the relevance of each text chunk for answering this question:

    QUESTION: {question}

    {len(chunks)} TEXT CHUNKS TO EVALUATE:
    {'-' * 40}
    {"".join(chunk_texts)}
    {'-' * 40}

    For each chunk, provide:
    1. A relevance score from 0-10 (where 10 is extremely relevant)
    2. A brief explanation of your scoring

    Return your evaluation as JSON with this structure for each chunk:
    {{
    "evaluations": [
        {{
        "chunk_id": 1,
        "score": 7,
        "explanation": "Brief explanation of why this chunk received this score"
        }},
        ...
    ]
    }}

    Important: Focus on how directly each chunk helps answer the specific question asked, considering:
    - Factual relevance to the question
    - Historical context provided
    - Specificity and detail level
    - Time period relevance
    """
        
        # System prompt for evaluation
        system_prompt = """You are an expert research assistant specializing in historical document analysis. 
    Your task is to evaluate text chunks from historical documents for their relevance to a specific research question.
    Be critical and precise in your evaluations. Prioritize chunks with direct factual relevance to the question. 
    You must return properly formatted JSON as specified in the instructions."""
        
        # Get LLM evaluation
        try:
            response = self.llm_service.generate_response(
                question=prompt,
                context="",
                model=model,
                system_prompt=system_prompt,
                openai_api_key=openai_api_key,
                temperature=0.2  # Low temperature for consistent evaluations
            )
            
            # Extract JSON from response
            eval_text = response.get('text', '')
            try:
                # Find JSON in the response
                import re
                json_match = re.search(r'```json\n(.*?)\n```|({[\s\S]*})', eval_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    evaluations = json.loads(json_str)
                else:
                    # Fallback - try to parse the whole response
                    evaluations = json.loads(eval_text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse evaluation JSON: {eval_text}")
                # Fallback: assign neutral scores
                evaluations = {"evaluations": [{"chunk_id": i+1, "score": 5, "explanation": "Error in evaluation"} for i in range(len(chunks))]}
            
            # Combine original chunks with evaluations
            result = []
            for i, chunk_data in enumerate(chunks):
                # Handle different chunk formats
                if len(chunk_data) == 2:
                    doc, vector_score = chunk_data
                else:
                    doc, vector_score, _ = chunk_data
                    
                # Find matching evaluation
                eval_entry = next((e for e in evaluations.get("evaluations", []) if e.get("chunk_id") == i+1), 
                            {"score": 5, "explanation": "No evaluation available"})
                
                # Normalize score to 0-1 range
                eval_score = float(eval_entry.get("score", 5)) / 10.0
                eval_explanation = eval_entry.get("explanation", "")
                
                result.append((doc, vector_score, eval_score, eval_explanation))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in chunk evaluation: {e}")
            # Fallback: return chunks with neutral eval scores
            return [(doc, score, 0.5, "Evaluation failed") for doc, score in chunks] if len(chunks[0]) == 2 else \
                [(doc, score, 0.5, "Evaluation failed") for doc, score, _ in chunks]