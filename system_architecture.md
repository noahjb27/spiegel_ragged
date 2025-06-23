# SPIEGEL RAG System - Technical Documentation

## Table of Contents

1. [System Overview](https://claude.ai/chat/dc92e404-e78e-447c-8f48-7ba55bac2dd6#system-overview)
2. [Architecture](https://claude.ai/chat/dc92e404-e78e-447c-8f48-7ba55bac2dd6#architecture)
3. [Core Components](https://claude.ai/chat/dc92e404-e78e-447c-8f48-7ba55bac2dd6#core-components)
4. [User Interface](https://claude.ai/chat/dc92e404-e78e-447c-8f48-7ba55bac2dd6#user-interface)
5. [Search Strategies](https://claude.ai/chat/dc92e404-e78e-447c-8f48-7ba55bac2dd6#search-strategies)
6. [Configuration](https://claude.ai/chat/dc92e404-e78e-447c-8f48-7ba55bac2dd6#configuration)
7. [API Integration](https://claude.ai/chat/dc92e404-e78e-447c-8f48-7ba55bac2dd6#api-integration)
8. [Data Flow](https://claude.ai/chat/dc92e404-e78e-447c-8f48-7ba55bac2dd6#data-flow)
9. [Deployment](https://claude.ai/chat/dc92e404-e78e-447c-8f48-7ba55bac2dd6#deployment)
10. [Development Guidelines](https://claude.ai/chat/dc92e404-e78e-447c-8f48-7ba55bac2dd6#development-guidelines)

## System Overview

The SPIEGEL RAG (Retrieval-Augmented Generation) System is a sophisticated research tool designed for analyzing Der Spiegel magazine archives from 1948-1979. It combines semantic search, time-aware analysis, and multiple LLM providers to enable comprehensive historical research.

### Key Features

* **Heuristik (Heuristics)** : Separated retrieval and analysis phases
* **LLM-Unterstützte Auswahl** : AI-assisted source evaluation with customizable prompts
* **Zeit-Intervall-Suche** : Time-windowed search for balanced temporal coverage
* **Interactive Source Selection** : Checkbox-based source selection with transfer functionality
* **Multi-LLM Support** : HU-LLM, DeepSeek R1, OpenAI GPT-4o, Google Gemini 2.5 Pro
* **Semantic Expansion** : FastText-based keyword expansion with corpus frequencies

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio UI     │───▶│   Search Engine  │───▶│  Vector Store   │
│                 │    │                  │    │   (ChromaDB)    │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Heuristik     │    │ • Strategy       │    │ • Embeddings    │
│ • Analyse       │    │   Pattern        │    │ • Similarity    │
│ • Source        │    │ • Time Windows   │    │   Search        │
│   Selection     │    │ • LLM Integration│    │ • Metadata      │
└─────────────────┘    └──────────────────┘    │   Filtering     │
                                               └─────────────────┘
                                                        │
                              ┌─────────────────────────┼─────────────────────────┐
                              │                         │                         │
                    ┌─────────▼────────┐    ┌───────▼───────┐    ┌──────▼──────────┐
                    │   LLM Service    │    │  Embedding    │    │  Archive Data   │
                    │                  │    │   Service     │    │                 │
                    │ • HU-LLM 1&3     │    │ • FastText    │    │ • Spiegel CSV   │
                    │ • DeepSeek R1    │    │ • Semantic    │    │ • 1948-1979     │
                    │ • OpenAI GPT-4o  │    │   Expansion   │    │ • Chunked Text  │
                    │ • Gemini 2.5 Pro │    │ • Frequencies │    │ • Metadata      │
                    └──────────────────┘    └───────────────┘    └─────────────────┘
```

### Component Architecture

```
src/
├── config/
│   └── settings.py          # Centralized configuration
├── core/
│   ├── engine.py           # Main RAG orchestrator
│   ├── vector_store.py     # ChromaDB interface
│   ├── llm_service.py      # Multi-LLM provider service
│   ├── embedding_service.py # FastText semantic expansion
│   ├── retrieval_agent.py  # LLM-assisted filtering
│   └── search/
│       ├── strategies.py   # Search strategy base classes
│       ├── agent_strategy.py # LLM-assisted search
│       └── enhanced_time_window_strategy.py
├── ui/
│   ├── app.py             # Main Gradio application
│   ├── components/        # UI component modules
│   ├── handlers/          # Event handlers
│   └── utils/             # UI utilities
├── tests/                 # Testing modules
└── utils/                 # General utilities
```

## Core Components

### 1. RAG Engine (`src/core/engine.py`)

The central orchestrator that coordinates search strategies and LLM analysis.

```python
class SpiegelRAG:
    def __init__(self):
        self.vector_store = ChromaDBInterface()
        self.llm_service = LLMService()
        self.embedding_service = WordEmbeddingService()
  
    def search(self, strategy, config, use_semantic_expansion=True) -> SearchResult
    def analyze(self, question, chunks, model, system_prompt, temperature) -> AnalysisResult
```

**Key Responsibilities:**

* Strategy pattern implementation for different search approaches
* Semantic expansion integration
* Result caching and state management
* Error handling and logging

### 2. Vector Store (`src/core/vector_store.py`)

ChromaDB interface for semantic search and metadata filtering.

```python
class ChromaDBInterface:
    def similarity_search(self, query, chunk_size, k, filter_dict, 
                         min_relevance_score, keywords, search_in, enforce_keywords)
    def build_metadata_filter(self, year_range, keywords, search_in)
```

**Features:**

* Multiple collection support (500, 2000, 3000 character chunks)
* Boolean keyword filtering with fallback parsing
* Metadata-based temporal filtering
* Relevance score thresholding
* Connection caching and error recovery

**ChromaDB Collections:**

* `recursive_chunks_3000_300_TH_cosine_nomic-embed-text`
* `recursive_chunks_2000_400_TH_cosine_nomic-embed-text`

### 3. LLM Service (`src/core/llm_service.py`)

Multi-provider LLM interface supporting various models and endpoints.

**Supported Providers:**

1. **HU-LLM** (Local Berlin network)
   * llm1-compute.cms.hu-berlin.de/v1/
   * llm3-compute.cms.hu-berlin.de/v1/
2. **DeepSeek R1** (Ollama)
   * 32B parameter reasoning model
3. **OpenAI** (External API)
   * GPT-4o with configurable parameters
4. **Google Gemini** (External API)
   * Gemini 2.5 Pro with large context window

```python
class LLMService:
    def generate_response(self, question, context, model, system_prompt, 
                         temperature, response_format=None)
    def health_check(self) -> Dict[str, Any]
    def get_available_models(self) -> List[str]
```

### 4. Embedding Service (`src/core/embedding_service.py`)

FastText-based semantic expansion and word analysis.

```python
class WordEmbeddingService:
    def find_similar_words(self, word, top_n=10) -> List[Dict]
    def parse_boolean_expression(self, expression) -> Dict[str, List[str]]
    def expand_search_terms(self, terms, expansion_factor=3) -> Dict
```

**Capabilities:**

* Semantic word similarity with FastText embeddings
* Corpus frequency analysis
* Boolean expression parsing (AND, OR, NOT)
* Query expansion with similarity thresholds

### 5. Search Strategies

#### Standard Search Strategy

* Direct vector similarity search
* Optional time-interval windowing
* Keyword filtering with boolean expressions
* Semantic expansion integration

#### LLM-Assisted Strategy (`src/core/search/agent_strategy.py`)

* Two-phase retrieval and evaluation
* Configurable evaluation prompts
* Time-windowed balanced retrieval
* Dual scoring (vector + LLM evaluation)
* Temperature-controlled evaluation consistency

```python
class TimeWindowedAgentStrategy:
    def __init__(self, llm_service, agent_config):
        self.llm_service = llm_service
        self.agent_config = agent_config
  
    def search(self, config, vector_store, progress_callback=None)
```

**Process Flow:**

1. **Initial Retrieval** : Broad collection (e.g., 50 chunks per time interval)
2. **LLM Evaluation** : Relevance scoring with detailed reasoning
3. **Iterative Filtering** : Multi-stage reduction (50→20→10)
4. **Final Selection** : Top-scored chunks with preserved reasoning

## User Interface

### Three-Tab Structure

#### 1. Heuristik Tab

 **Purpose** : Source discovery and selection

* **Search Method Selection** : Standard vs LLM-Assisted
* **Search Configuration** : Time periods, chunking, keywords
* **Interactive Results Display** : Checkbox-based selection
* **Source Transfer** : Explicit transfer to analysis phase

#### 2. Analyse Tab

 **Purpose** : Question answering and analysis

* **Transferred Sources Display** : Review selected sources
* **User Prompt Configuration** : Research question formulation
* **LLM Selection** : Model and parameter configuration
* **System Prompt Management** : Template-based prompt customization

#### 3. Info Tab

 **Purpose** : Documentation and system information

### Key UI Components

#### Interactive Chunk Selection (`src/ui/components/retrieved_chunks_display.py`)

```javascript
// JavaScript handles visual state
function updateVisualSummary() { /* checkbox management */ }
function confirmCurrentSelection() { /* state confirmation */ }

// Python handles confirmed state
def confirm_selection(js_selection_json, available_chunks) -> tuple
def transfer_chunks_to_analysis(available_chunks, confirmed_selection) -> tuple
```

**State Management:**

1. **Visual State** : JavaScript checkbox interactions
2. **Confirmation State** : Explicit user confirmation
3. **Transfer State** : Python-managed analysis state

## Search Strategies

### 1. Standard Search (Heuristik)

**Configuration Options:**

* **Chunking Size** : 500, 2000, or 3000 characters
* **Time Range** : 1948-1979 (adjustable)
* **Result Count** : 1-50 sources total OR per time-interval
* **Keyword Filtering** : Boolean expressions (AND, OR, NOT)
* **Semantic Expansion** : FastText-based term expansion
* **Time-Interval Search** : Balanced temporal distribution

**Search Flow:**

```python
# 1. Configure search
config = SearchConfig(
    content_description="Berichterstattung über die Berliner Mauer",
    year_range=(1960, 1970),
    chunk_size=3000,
    keywords="berlin AND mauer",
    search_fields=["Text", "Artikeltitel"],
    top_k=10
)

# 2. Execute strategy
strategy = StandardSearchStrategy() # or TimeWindowSearchStrategy()
result = engine.search(strategy, config, use_semantic_expansion=True)
```

### 2. LLM-Assisted Selection (LLM-Unterstützte Auswahl)

**Enhanced Features:**

* **Time-Interval Integration** : Default enabled for balanced coverage
* **Evaluation Prompts** : Customizable assessment criteria
* **Temperature Control** : Consistency vs. creativity in evaluation
* **Dual Scoring** : Vector similarity + LLM evaluation scores
* **Transparent Reasoning** : Detailed evaluation explanations

**Configuration Parameters:**

```python
agent_config = AgentSearchConfig(
    use_time_windows=True,
    time_window_size=5,  # years
    chunks_per_window_initial=50,
    chunks_per_window_final=20,
    agent_model="hu-llm3",
    evaluation_temperature=0.2,
    min_retrieval_relevance_score=0.25
)
```

**Evaluation Process:**

1. **Initial Retrieval** : Broad collection per time interval
2. **Batch Evaluation** : Process chunks in optimal token-sized batches
3. **Scoring Criteria** : Relevance scoring (0-10) with reasoning
4. **Iterative Filtering** : Progressive reduction with top-scored retention
5. **Final Assembly** : Time-balanced final selection

## Configuration

### Environment Variables (`.env`)

```bash
# LLM API Settings
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key

# HU LLM Endpoints
HU_LLM1_API_URL=https://llm1-compute.cms.hu-berlin.de/v1/
HU_LLM3_API_URL=https://llm3-compute.cms.hu-berlin.de/v1/

# ChromaDB Settings
CHROMA_DB_HOST=dighist.geschichte.hu-berlin.de
CHROMA_DB_PORT=8000
CHROMA_DB_SSL=true

# Ollama Settings (DeepSeek R1)
OLLAMA_BASE_URL=https://dighist.geschichte.hu-berlin.de:11434
DEEPSEEK_R1_MODEL_NAME=deepseek-r1:32b

# Model Settings
WORD_EMBEDDING_MODEL_PATH=./models/fasttext_model_spiegel_corpus_neu_50epochs_2.model
DEFAULT_CHUNK_SIZE=3000
DEFAULT_LLM_MODEL=hu-llm3
```

### System Prompts (`src/config/settings.py`)

**Default Analysis Prompt:**

```python
SYSTEM_PROMPTS = {
    "default": """Du bist ein erfahrener Historiker mit Expertise in der kritischen Auswertung von SPIEGEL-Artikeln aus den Jahren 1948-1979.

**Hauptaufgabe**: Beantworte die Forschungsfrage präzise und wissenschaftlich fundiert basierend ausschließlich auf den bereitgestellten Textauszügen.

**Methodik**:
* **Quellentreue**: Nutze ausschließlich die bereitgestellten Textauszüge als Grundlage
* **Wissenschaftliche Präzision**: Formuliere analytisch und differenziert
* **Vollständige Integration**: Berücksichtige möglichst viele relevante Textauszüge
* **Transparente Belege**: Verweise präzise mit [Datum, Artikeltitel]
```

**LLM-Assisted Evaluation Prompts:**

```python
LLM_ASSISTED_SYSTEM_PROMPTS = {
    "standard_evaluation": """Du bewertest Textabschnitte aus SPIEGEL-Artikeln (1948-1979) für historische Forschung.

**Aufgabe**: Analysiere zunächst jeden Textabschnitt ausführlich im Hinblick auf seine Relevanz für den user retrieval Query. Führe eine differenzierte Argumentation durch, bevor du eine Bewertung abgibst.

**Bewertungsskala**:
- 9-10: Direkt relevant mit substanziellen Informationen
- 7-8: Stark relevant mit wichtigem Kontext
- 5-6: Mäßig relevant mit ergänzenden Aspekten
- 3-4: Schwach relevant mit entferntem Bezug
- 0-2: Nicht relevant für die Fragestellung
```

## API Integration

### External APIs

#### OpenAI Integration

```python
def _generate_openai_response(self, client_info, prompt, system_prompt, 
                             temperature, model, response_format=None):
    request_params = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "model": model_id,
        "temperature": temperature
    }
  
    if response_format:
        request_params["response_format"] = response_format
```

#### Gemini Integration

```python
def _generate_gemini_response(self, client_info, prompt, system_prompt, 
                             temperature, model):
    # Combine system prompt and user prompt for Gemini
    full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
  
    generation_config = {"temperature": temperature}
    model_instance = genai.GenerativeModel(model_id)
  
    response = model_instance.generate_content(
        full_prompt, generation_config=generation_config
    )
```

#### DeepSeek R1 (Ollama) Integration

```python
def _generate_ollama_response(self, client_info, prompt, system_prompt, 
                             temperature, model):
    url = f"{endpoint}/api/chat"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": temperature}
    }
  
    response = requests.post(url, json=payload, timeout=120)
```

### Internal APIs (HU-LLM)

* OpenAI-compatible endpoints
* No authentication required (network-based access control)
* Standard chat completion format

## Data Flow

### Search Phase Data Flow

```
User Input → Search Config → Strategy Selection → Vector Store Query
     ↓
Semantic Expansion ← FastText Embeddings ← Keywords (optional)
     ↓
ChromaDB Search → Relevance Filtering → Time Window Grouping
     ↓
LLM Evaluation (if agent mode) → Score Ranking → Result Assembly
     ↓
UI Display → Chunk Selection → Transfer to Analysis
```

### Analysis Phase Data Flow

```
Transferred Chunks → Document Conversion → Context Assembly
     ↓
User Prompt + System Prompt + Context → LLM Service
     ↓
Model Selection → Provider Routing → Response Generation
     ↓
Answer Assembly → Citation Formatting → UI Display
     ↓
Download Options (TXT, JSON, CSV)
```

### State Management Flow

```
Search Results → Available Chunks State
     ↓
Visual Checkbox Selection → JavaScript State
     ↓
Confirm Selection Button → Confirmed Selection State  
     ↓
Transfer Button → Transferred Chunks State
     ↓
Analysis Phase → Final Results
```

## Deployment

### Prerequisites

* Python 3.8+
* Access to HU Berlin network (Eduroam/VPN) for local services
* Optional: OpenAI/Gemini API keys for external LLMs

### Installation Steps

1. **Clone Repository**
2. **Create Virtual Environment**
3. **Install Dependencies** : `pip install -r requirements.txt`
4. **Download Models** : Run `python models/model_data_import.py`
5. **Download Data** (optional): Run `python data/spiegel_data_import.py`
6. **Configure Environment** : Copy `.env.example` to `.env` and customize
7. **Launch Application** : `python src/ui/app.py`

### Network Configuration

* **ChromaDB** : `dighist.geschichte.hu-berlin.de:8000`
* **Ollama** : `https://dighist.geschichte.hu-berlin.de:11434`
* **HU-LLM** : Multiple endpoints on `cms.hu-berlin.de`

### Resource Requirements

* **Memory** : 8GB+ recommended for FastText embeddings
* **Storage** : 2GB+ for models and cached data
* **Network** : Stable connection to HU services

## Development Guidelines

### Code Structure Principles

1. **Strategy Pattern** : All search methods implement `SearchStrategy`
2. **Service Layer** : Clear separation between UI, business logic, and data access
3. **German Terminology** : Consistent use of updated German terms
4. **Error Handling** : Comprehensive try-catch with graceful degradation
5. **Type Hints** : Full type annotation for better IDE support

### Testing Strategy

```python
# Component tests
python src/utils/component_test.py

# Network integration tests  
python src/tests/deepseek_integration.py

# Filter robustness tests
python src/tests/filter_test.py
```

### Adding New Search Strategies

```python
class CustomSearchStrategy(SearchStrategy):
    def search(self, config: SearchConfig, vector_store: Any, 
              progress_callback: Optional[Callable] = None) -> SearchResult:
        # Implementation
        return SearchResult(chunks=results, metadata=metadata)
```

### Adding New LLM Providers

```python
def _generate_custom_response(self, client_info, prompt, system_prompt, 
                             temperature, model):
    # Provider-specific implementation
    return {
        "text": response_text,
        "model": model,
        "provider": "custom",
        "metadata": {}
    }
```

### UI Component Guidelines

* Use German terminology consistently
* Implement proper accordion state management
* Include comprehensive error messages
* Support download functionality
* Maintain accessibility standards

This documentation provides the foundation for understanding, maintaining, and extending the SPIEGEL RAG system while following established patterns and conventions.
