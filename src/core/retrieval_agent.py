# src/core/retrieval_agent_improved.py
"""
Enhanced RetrievalAgent with robust JSON handling and structured evaluation.
Drop-in replacement for the existing RetrievalAgent.
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
    Enhanced RetrievalAgent with robust structured evaluation.
    Direct replacement for the original RetrievalAgent.
    """
    
    def __init__(self, vector_store: ChromaDBInterface, llm_service: LLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.evaluation_adapter = TypeAdapter(EvaluationResponse)
        logger.info("Initialized ImprovedRetrievalAgent with structured evaluation")
    
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
        Enhanced retrieval and refinement with structured evaluation.
        Maintains same interface as original for compatibility.
        """
        start_time = time.time()
        stage_times = []
        stage_results = []
        
        # Use defaults if not provided
        chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        year_range = year_range or [settings.MIN_YEAR, settings.MAX_YEAR]
        
        logger.info(f"Starting enhanced retrieval agent with question: '{question}'")
        logger.info(f"Initial retrieval count: {initial_retrieval_count}, Filter stages: {filter_stages}")
        
        # Step 1: Initial retrieval (same as original)
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
        
        # Step 2: Enhanced iterative refinement
        for i, target_count in enumerate(filter_stages):
            stage_start = time.time()
            
            if len(current_chunks) <= target_count:
                logger.info(f"Skipping filter stage {i+1} as we already have {len(current_chunks)} ≤ {target_count} chunks")
                continue
                
            logger.info(f"Starting enhanced filter stage {i+1}: {len(current_chunks)} → {target_count} chunks")
            
            # Use enhanced evaluation
            evaluated_chunks = self._evaluate_chunks_structured(
                question=question,
                chunks=current_chunks,
                model=model,
                openai_api_key=openai_api_key,
                batch_size=min(20, len(current_chunks))  # Smaller batches for better reliability
            )
            
            # Sort by evaluation score and keep top chunks
            evaluated_chunks.sort(key=lambda x: x[2], reverse=True)
            current_chunks = [(doc, score, eval_text) for doc, score, eval_score, eval_text in evaluated_chunks[:target_count]]
            
            stage_time = time.time() - stage_start
            stage_times.append((f"Enhanced Filter Stage {i+1}", stage_time))
            stage_results.append(len(current_chunks))
            
            logger.info(f"Completed enhanced filter stage {i+1}: reduced to {len(current_chunks)} chunks in {stage_time:.2f}s")
        
        # Return final results
        total_time = time.time() - start_time
        logger.info(f"Enhanced retrieval agent completed in {total_time:.2f}s")
        
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
            "enhancement": "structured_evaluation"
        }
        
        return final_chunks, metadata
    
    def _evaluate_chunks_structured(
        self,
        question: str,
        chunks: List,
        model: str = "hu-llm",
        openai_api_key: Optional[str] = None,
        batch_size: int = 20
    ) -> List[Tuple[Document, float, float, str]]:
        """
        Enhanced chunk evaluation with structured JSON responses.
        Replacement for the original _evaluate_chunks method.
        """
        if not chunks:
            return []
            
        all_evaluated_chunks = []
        
        # Process chunks in batches
        chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        logger.info(f"Evaluating chunks in {len(chunk_batches)} batches using structured approach")
        
        for batch_idx, batch in enumerate(chunk_batches):
            logger.info(f"Processing structured evaluation batch {batch_idx + 1}/{len(chunk_batches)}")
            batch_result = self._evaluate_batch_structured(question, batch, model, openai_api_key)
            all_evaluated_chunks.extend(batch_result)
            
        return all_evaluated_chunks
    
    def _evaluate_batch_structured(
        self,
        question: str,
        chunks: List,
        model: str,
        openai_api_key: Optional[str] = None,
        max_retries: int = 3
    ) -> List[Tuple[Document, float, float, str]]:
        """
        Evaluate a batch of chunks using structured JSON output.
        Enhanced version with robust error handling and schema validation.
        """
        # Prepare chunk data for evaluation
        chunk_summaries = []
        for i, chunk_data in enumerate(chunks):
            if len(chunk_data) == 2:
                doc, score = chunk_data
            else:
                doc, score, _ = chunk_data
            
            # Create structured chunk summary
            chunk_summaries.append({
                "id": i + 1,
                "title": doc.metadata.get('Artikeltitel', 'No title')[:100],
                "date": doc.metadata.get('Datum', 'Unknown'),
                "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            })
        
        # Prepare structured messages
        messages = [
            {
                "role": "system",
                "content": """Du bist ein Experte für die Bewertung historischer Dokumente. 
                Bewerte jeden Textabschnitt für seine Relevanz zur Forschungsfrage auf einer Skala von 0-10.
                Berücksichtige: faktische Relevanz, historischen Kontext, Spezifität und zeitliche Relevanz.
                Antworte AUSSCHLIESSLICH mit gültigem JSON im angegebenen Format.
                Sei kritisch - die meisten Chunks sollten 3-7 bekommen, nur außergewöhnliche erhalten 8-10."""
            },
            {
                "role": "user",
                "content": f"""Forschungsfrage: {question}

                Bewerte diese {len(chunk_summaries)} Textabschnitte:

                {json.dumps(chunk_summaries, indent=2, ensure_ascii=False)}

                Gib für jeden Chunk eine strukturierte Bewertung zurück. Format:
                {{
                "evaluations": [
                    {{
                    "chunk_id": 1,
                    "score": 7.5,
                    "explanation": "Klare Begründung für die Bewertung",
                    "confidence": "high"
                    }}
                ],
                "batch_summary": "Kurze Zusammenfassung des Bewertungsansatzes"
                }}"""
            }
        ]
        
        # Attempt structured evaluation with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Structured evaluation attempt {attempt + 1}")
                
                # Generate response with appropriate method
                if model.startswith("gpt"):
                    # OpenAI with JSON mode
                    response = self.llm_service.openai_client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=0.2,
                        response_format={"type": "json_object"}
                    )
                    response_text = response.choices[0].message.content
                else:
                    # HU-LLM or other models with schema enforcement
                    response = self.llm_service.hu_llm_client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=0.2,
                        format=self.evaluation_adapter.json_schema() if hasattr(self.evaluation_adapter, 'json_schema') else ""
                    )
                    response_text = response.choices[0].message.content
                
                # Extract and validate JSON
                validated_response = self._extract_and_validate_json(response_text)
                
                if validated_response:
                    logger.info(f"Successfully parsed structured evaluation for batch")
                    return self._process_structured_response(chunks, validated_response)
                else:
                    raise ValueError("Failed to extract valid JSON from response")
                
            except Exception as e:
                logger.warning(f"Structured evaluation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All structured evaluation attempts failed, using fallback")
                    return self._fallback_evaluation(chunks)
        
        return self._fallback_evaluation(chunks)
    
    def _extract_and_validate_json(self, response_text: str) -> Optional[EvaluationResponse]:
        """
        Extract and validate JSON from LLM response using multiple methods.
        """
        try:
            # Method 1: Direct JSON parsing (cleanest response)
            try:
                json_data = json.loads(response_text.strip())
                return self.evaluation_adapter.validate_python(json_data)
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Method 2: Use jsonfinder for robust extraction
            if jsonfinder:
                json_objects = [obj for _, _, obj in jsonfinder.jsonfinder(response_text) if obj is not None]
                if json_objects:
                    return self.evaluation_adapter.validate_python(json_objects[0])
            
            # Method 3: Regex extraction as fallback
            json_match = re.search(r'```json\s*\n(.*?)\n\s*```|(\{[\s\S]*\})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                json_data = json.loads(json_str)
                return self.evaluation_adapter.validate_python(json_data)
            
            logger.warning("No valid JSON found in response")
            return None
            
        except Exception as e:
            logger.error(f"JSON extraction failed: {e}")
            return None
    
    def _process_structured_response(
        self, 
        chunks: List, 
        validated_response: EvaluationResponse
    ) -> List[Tuple[Document, float, float, str]]:
        """Process validated structured response into expected format"""
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
        
        logger.info(f"Processed {len(results)} structured evaluations")
        return results
    
    def _fallback_evaluation(self, chunks: List) -> List[Tuple[Document, float, float, str]]:
        """Fallback evaluation when structured approach fails"""
        logger.warning("Using fallback evaluation due to structured evaluation failure")
        
        results = []
        for chunk_data in chunks:
            if len(chunk_data) == 2:
                doc, vector_score = chunk_data
            else:
                doc, vector_score, _ = chunk_data
            
            # Assign neutral evaluation with indication of fallback
            fallback_score = 0.5
            eval_text = "Fallback evaluation used (structured evaluation failed)"
            
            results.append((doc, vector_score, fallback_score, eval_text))
        
        return results


# Drop-in replacement function for easy migration
def create_improved_retrieval_agent(vector_store: ChromaDBInterface, llm_service: LLMService):
    """Factory function to create improved retrieval agent"""
    return RetrievalAgent(vector_store, llm_service)