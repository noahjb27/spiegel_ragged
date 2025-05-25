# src/core/retrieval_agent.py - FIXED VERSION
"""
Fixed RetrievalAgent with proper JSON handling that works with both HU-LLM and OpenAI.
The main fix is removing the unsupported 'format' parameter and using proper prompting.
"""
import logging
import json
import time
import re
from typing import Dict, List, Optional, Tuple, Any, Union

from pydantic import BaseModel, Field, field_validator
from pydantic import TypeAdapter
from langchain.docstore.document import Document

try:
    import jsonfinder
except ImportError:
    print("Please install jsonfinder: pip install jsonfinder")
    jsonfinder = None

from src.core.vector_store import ChromaDBInterface
from src.core.llm_service import LLMService
from src.config import settings

logger = logging.getLogger(__name__)


# Pydantic models for structured evaluation
class ChunkEvaluation(BaseModel):
    """Individual chunk evaluation with validation"""
    chunk_id: int = Field(..., ge=1, description="1-based chunk identifier")
    score: float = Field(..., ge=0.0, le=10.0, description="Relevance score 0-10")
    explanation: str = Field(..., min_length=10, max_length=200, description="Brief evaluation reasoning")
    confidence: str = Field(default="medium", description="Confidence in evaluation (high/medium/low)")
    
    @field_validator('score')
    @classmethod
    def round_score(cls, v):
        return round(v, 1)
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        allowed = ['high', 'medium', 'low']
        if v.lower() not in allowed:
            return 'medium'
        return v.lower()

class EvaluationResponse(BaseModel):
    """Complete evaluation response with validation"""
    evaluations: List[ChunkEvaluation]
    batch_summary: str = Field(default="", description="Brief summary of evaluation approach")
    
    @field_validator('evaluations')
    @classmethod
    def validate_evaluations(cls, v):
        if not v:
            raise ValueError("At least one evaluation required")
        return v

class RetrievalAgent:
    """
    Fixed RetrievalAgent with proper JSON handling for both HU-LLM and OpenAI.
    Main fix: removed unsupported 'format' parameter and improved prompting.
    """
    
    def __init__(self, vector_store: ChromaDBInterface, llm_service: LLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.evaluation_adapter = TypeAdapter(EvaluationResponse)
        logger.info("Initialized Fixed RetrievalAgent with proper JSON handling")
    
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
    ) -> Tuple[List[Tuple[Document, float, float, str]], Dict[str, Any]]:
        """
        Fixed retrieval and refinement with proper JSON handling.
        """
        start_time = time.time()
        stage_times = []
        stage_results = []
        
        # Use defaults if not provided
        chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        year_range = year_range or [settings.MIN_YEAR, settings.MAX_YEAR]
        
        logger.info(f"Starting fixed retrieval agent with question: '{question}'")
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
        
        # Step 2: Fixed iterative refinement
        for i, target_count in enumerate(filter_stages):
            stage_start = time.time()
            
            if len(current_chunks) <= target_count:
                logger.info(f"Skipping filter stage {i+1} as we already have {len(current_chunks)} ≤ {target_count} chunks")
                continue
                
            logger.info(f"Starting fixed filter stage {i+1}: {len(current_chunks)} → {target_count} chunks")
            
            # Use fixed evaluation
            evaluated_chunks = self._evaluate_chunks_fixed(
                question=question,
                chunks=current_chunks,
                model=model,
                openai_api_key=openai_api_key,
                batch_size=min(15, len(current_chunks))  # Smaller batches for reliability
            )
            
            # Sort by evaluation score and keep top chunks
            evaluated_chunks.sort(key=lambda x: x[2], reverse=True)
            current_chunks = [(doc, score, eval_text) for doc, score, eval_score, eval_text in evaluated_chunks[:target_count]]
            
            stage_time = time.time() - stage_start
            stage_times.append((f"Fixed Filter Stage {i+1}", stage_time))
            stage_results.append(len(current_chunks))
            
            logger.info(f"Completed fixed filter stage {i+1}: reduced to {len(current_chunks)} chunks in {stage_time:.2f}s")
        
        # Return final results
        total_time = time.time() - start_time
        logger.info(f"Fixed retrieval agent completed in {total_time:.2f}s")
        
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
            "final_chunk_count": len(final_chunks),
            "enhancement": "fixed_json_handling"
        }
        
        return final_chunks, metadata
    
    def _evaluate_chunks_fixed(
        self,
        question: str,
        chunks: List,
        model: str = "hu-llm",
        openai_api_key: Optional[str] = None,
        batch_size: int = 15
    ) -> List[Tuple[Document, float, float, str]]:
        """
        Fixed chunk evaluation that works with both HU-LLM and OpenAI.
        Main fix: removed unsupported format parameter.
        """
        if not chunks:
            return []
            
        all_evaluated_chunks = []
        
        # Process chunks in smaller batches for reliability
        chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        logger.info(f"Evaluating chunks in {len(chunk_batches)} batches using fixed approach")
        
        for batch_idx, batch in enumerate(chunk_batches):
            logger.info(f"Processing fixed evaluation batch {batch_idx + 1}/{len(chunk_batches)}")
            batch_result = self._evaluate_batch_fixed(question, batch, model, openai_api_key)
            all_evaluated_chunks.extend(batch_result)
            
        return all_evaluated_chunks
    
    def _evaluate_batch_fixed(
        self,
        question: str,
        chunks: List,
        model: str,
        openai_api_key: Optional[str] = None,
        max_retries: int = 2
    ) -> List[Tuple[Document, float, float, str]]:
        """
        Fixed batch evaluation that works with both HU-LLM and OpenAI.
        Key fix: removed format parameter and improved prompting.
        """
        # Prepare chunk data for evaluation
        chunk_summaries = []
        for i, chunk_data in enumerate(chunks):
            if len(chunk_data) == 2:
                doc, score = chunk_data
            else:
                doc, score, _ = chunk_data
            
            chunk_summaries.append({
                "id": i + 1,
                "title": doc.metadata.get('Artikeltitel', 'No title')[:100],
                "date": doc.metadata.get('Datum', 'Unknown'),
                "content_preview": doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
            })
        
        # Use LLM service's generate_response method instead of direct API calls
        system_prompt = """Du bist ein Experte für die Bewertung historischer Dokumente. 
Bewerte jeden Textabschnitt für seine Relevanz zur Forschungsfrage auf einer Skala von 0-10.
Berücksichtige: faktische Relevanz, historischen Kontext, Spezifität und zeitliche Relevanz.

WICHTIG: Antworte AUSSCHLIESSLICH mit gültigem JSON im folgenden Format:
{
  "evaluations": [
    {
      "chunk_id": 1,
      "score": 7.5,
      "explanation": "Kurze Begründung der Bewertung",
      "confidence": "high"
    }
  ],
  "batch_summary": "Kurze Zusammenfassung des Bewertungsansatzes"
}

Sei kritisch - die meisten Chunks sollten 3-7 bekommen, nur außergewöhnliche erhalten 8-10."""

        user_prompt = f"""Forschungsfrage: {question}

Bewerte diese {len(chunk_summaries)} Textabschnitte:

{json.dumps(chunk_summaries, indent=2, ensure_ascii=False)}

Gib für jeden Chunk eine strukturierte Bewertung zurück."""

        # Attempt evaluation with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Fixed evaluation attempt {attempt + 1}")
                
                # Use the LLM service's generate_response method
                response = self.llm_service.generate_response(
                    question=user_prompt,
                    context="",  # No additional context needed
                    model=model,
                    system_prompt=system_prompt,
                    temperature=0.2,
                    max_tokens=2000,  # Ensure enough tokens for JSON response
                    openai_api_key=openai_api_key,
                    response_format={"type": "json_object"} if model.startswith("gpt") else None
                )
                
                response_text = response.get('text', '')
                
                # Extract and validate JSON
                validated_response = self._extract_and_validate_json_fixed(response_text)
                
                if validated_response:
                    logger.info(f"Successfully parsed fixed evaluation for batch")
                    return self._process_fixed_response(chunks, validated_response)
                else:
                    raise ValueError("Failed to extract valid JSON from response")
                
            except Exception as e:
                logger.warning(f"Fixed evaluation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All fixed evaluation attempts failed, using simple evaluation")
                    return self._simple_evaluation(chunks, question, model, openai_api_key)
        
        return self._simple_evaluation(chunks, question, model, openai_api_key)
    
    def _extract_and_validate_json_fixed(self, response_text: str) -> Optional[EvaluationResponse]:
        """
        Fixed JSON extraction that's more robust.
        """
        try:
            # Method 1: Direct JSON parsing
            try:
                json_data = json.loads(response_text.strip())
                return self.evaluation_adapter.validate_python(json_data)
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Method 2: Use jsonfinder for robust extraction
            if jsonfinder:
                try:
                    json_objects = [obj for _, _, obj in jsonfinder.jsonfinder(response_text) if obj is not None]
                    if json_objects:
                        return self.evaluation_adapter.validate_python(json_objects[0])
                except Exception:
                    pass
            
            # Method 3: Regex extraction as fallback
            json_patterns = [
                r'```json\s*\n(.*?)\n\s*```',
                r'```\s*\n(\{[\s\S]*?\})\s*\n```',
                r'(\{[\s\S]*"evaluations"[\s\S]*\})'
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    try:
                        json_str = match.group(1)
                        json_data = json.loads(json_str)
                        return self.evaluation_adapter.validate_python(json_data)
                    except Exception:
                        continue
            
            logger.warning("No valid JSON found in response")
            return None
            
        except Exception as e:
            logger.error(f"JSON extraction failed: {e}")
            return None
    
    def _process_fixed_response(
        self, 
        chunks: List, 
        validated_response: EvaluationResponse
    ) -> List[Tuple[Document, float, float, str]]:
        """Process validated response into expected format"""
        results = []
        
        for chunk_data, evaluation in zip(chunks, validated_response.evaluations):
            # Handle different chunk formats
            if len(chunk_data) == 2:
                doc, vector_score = chunk_data
            else:
                doc, vector_score, _ = chunk_data
            
            # Process evaluation
            eval_score = evaluation.score / 10.0  # Normalize to 0-1
            
            # Apply confidence weighting
            confidence_multipliers = {"high": 1.0, "medium": 0.95, "low": 0.85}
            confidence_mult = confidence_multipliers.get(evaluation.confidence, 0.95)
            weighted_score = eval_score * confidence_mult
            
            # Create detailed evaluation text
            eval_text = f"Score: {evaluation.score}/10 (confidence: {evaluation.confidence}). {evaluation.explanation}"
            
            results.append((doc, vector_score, weighted_score, eval_text))
        
        logger.info(f"Processed {len(results)} fixed evaluations with meaningful scores")
        return results
    
    def _simple_evaluation(
        self,
        chunks: List,
        question: str,
        model: str,
        openai_api_key: Optional[str] = None
    ) -> List[Tuple[Document, float, float, str]]:
        """
        Simple evaluation fallback that still provides meaningful scores.
        """
        logger.info("Using simple evaluation fallback with LLM scoring")
        
        results = []
        
        for chunk_data in chunks:
            if len(chunk_data) == 2:
                doc, vector_score = chunk_data
            else:
                doc, vector_score, _ = chunk_data
            
            try:
                # Create a simple relevance prompt
                simple_prompt = f"""Bewerte die Relevanz dieses Textabschnitts für die Frage "{question}" auf einer Skala von 0-10.
Antworte nur mit einer Zahl zwischen 0 und 10.

Text: {doc.page_content[:500]}...

Score (0-10):"""
                
                response = self.llm_service.generate_response(
                    question=simple_prompt,
                    context="",
                    model=model,
                    system_prompt="Du bist ein Experte für Dokumentenbewertung. Antworte nur mit einer Zahl.",
                    temperature=0.1,
                    max_tokens=5,
                    openai_api_key=openai_api_key
                )
                
                # Extract score from response
                score_text = response.get('text', '5').strip()
                try:
                    simple_score = float(re.findall(r'\d+\.?\d*', score_text)[0])
                    simple_score = max(0, min(10, simple_score))  # Clamp to 0-10
                    eval_score = simple_score / 10.0  # Normalize to 0-1
                except (ValueError, IndexError):
                    eval_score = 0.5  # Default fallback
                
                eval_text = f"Simple evaluation score: {simple_score}/10"
                
            except Exception as e:
                logger.warning(f"Simple evaluation failed for chunk: {e}")
                eval_score = 0.5
                eval_text = "Simple evaluation failed, using default score"
            
            results.append((doc, vector_score, eval_score, eval_text))
        
        logger.info(f"Completed simple evaluation for {len(results)} chunks")
        return results