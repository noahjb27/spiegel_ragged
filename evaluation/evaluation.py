#!/usr/bin/env python3
"""
Redesigned evaluation script:
- Entnazifizierung: Retrieval mechanics analysis
- Homosexualität: Answer quality evaluation
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
import numpy as np
from itertools import combinations

def load_results(filepath):
    """Load JSON results from search"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
        return {'chunks': []}

def normalize_chunk_data(results):
    """Normalize chunk data from different result formats"""
    normalized_chunks = []
    
    # Handle LLM-assisted results (evaluations array)
    if 'evaluations' in results.get('heuristik_metadata', {}):
        for eval_item in results['heuristik_metadata']['evaluations']:
            # Use original_llm_score (0-10) if available, otherwise llm_score (0-1)
            llm_score = eval_item.get('original_llm_score', eval_item.get('llm_score', 0))
            if llm_score <= 1.0:  # If it's normalized, convert back to 0-10 scale
                llm_score = llm_score * 10
                
            normalized_chunks.append({
                'title': eval_item.get('title', 'No title'),
                'date': eval_item.get('date', 'No date'),
                'vector_score': eval_item.get('vector_score', 0),
                'llm_score': llm_score,
                'original_llm_score': eval_item.get('original_llm_score', 0),
                'content': eval_item.get('evaluation', ''),  # Use evaluation text as content
                'window': eval_item.get('window', ''),
                'source_type': 'llm_assisted'
            })
    
    # Handle standard search results (chunks array)
    elif 'chunks' in results:
        for chunk in results['chunks']:
            metadata = chunk.get('metadata', {})
            normalized_chunks.append({
                'title': metadata.get('titel', 'No title'),
                'date': metadata.get('datum', 'No date'),
                'vector_score': chunk.get('relevance_score', 0),
                'llm_score': 0,  # No LLM score in standard search
                'original_llm_score': 0,
                'content': chunk.get('content', '')[:200],  # Use content preview
                'window': '',
                'source_type': 'standard'
            })
    
    return normalized_chunks

def get_chunk_signature(chunk):
    """Create unique identifier for chunk comparison"""
    # Use title + date for matching (more reliable than content)
    title = chunk.get('title', 'unknown')
    date = chunk.get('date', 'unknown')
    return f"{title}_{date}_{hash(str(chunk.get('content', '')))[:4]}"

def analyze_retrieval_overlap(results_dict, case_name="Case"):
    """Analyze chunk overlap across different retrieval methods"""
    
    print(f"\n{'='*60}")
    print(f"RETRIEVAL OVERLAP ANALYSIS: {case_name}")
    print(f"{'='*60}")
    
    # Normalize and get chunk signatures for each method
    method_chunks = {}
    method_scores = {}
    
    for method_name, results in results_dict.items():
        # Normalize the chunk data
        normalized_chunks = normalize_chunk_data(results)
        signatures = [get_chunk_signature(chunk) for chunk in normalized_chunks]
        method_chunks[method_name] = set(signatures)
        
        # Store scores for overlap analysis
        method_scores[method_name] = {}
        for chunk in normalized_chunks:
            sig = get_chunk_signature(chunk)
            method_scores[method_name][sig] = {
                'vector': chunk.get('vector_score', 0),
                'llm': chunk.get('llm_score', 0),
                'title': chunk.get('title', 'No title'),
                'date': chunk.get('date', 'No date'),
                'source_type': chunk.get('source_type', 'unknown')
            }
        
        print(f"\n{method_name}: {len(normalized_chunks)} chunks")
    
    # Calculate pairwise overlaps
    overlap_matrix = {}
    for method1, method2 in combinations(method_chunks.keys(), 2):
        overlap = len(method_chunks[method1] & method_chunks[method2])
        total_unique = len(method_chunks[method1] | method_chunks[method2])
        overlap_pct = (overlap / len(method_chunks[method1])) * 100 if method_chunks[method1] else 0
        
        print(f"\n{method1} vs {method2}:")
        print(f"  Overlapping chunks: {overlap}")
        print(f"  {method1} unique: {len(method_chunks[method1] - method_chunks[method2])}")
        print(f"  {method2} unique: {len(method_chunks[method2] - method_chunks[method1])}")
        print(f"  Overlap percentage: {overlap_pct:.1f}%")
        
        overlap_matrix[f"{method1}_vs_{method2}"] = {
            'overlap': overlap,
            'overlap_pct': overlap_pct,
            'unique_1': len(method_chunks[method1] - method_chunks[method2]),
            'unique_2': len(method_chunks[method2] - method_chunks[method1])
        }
    
    # Find core chunks (appear in multiple methods)
    all_chunks = set.union(*method_chunks.values()) if method_chunks else set()
    chunk_frequency = Counter()
    for chunk_sig in all_chunks:
        for method_name, chunks in method_chunks.items():
            if chunk_sig in chunks:
                chunk_frequency[chunk_sig] += 1
    
    print(f"\nCHUNK FREQUENCY ANALYSIS:")
    freq_dist = Counter(chunk_frequency.values())
    for freq, count in sorted(freq_dist.items(), reverse=True):
        print(f"  Chunks in {freq} methods: {count}")
    
    # Analyze core chunks (appear in 3+ methods)
    if len(method_chunks) >= 3:
        core_chunks = [chunk for chunk, freq in chunk_frequency.items() if freq >= 3]
        print(f"\nCORE CHUNKS (appear in 3+ methods): {len(core_chunks)}")
        
        if core_chunks:
            print("Sample core chunks:")
            for chunk_sig in core_chunks[:3]:
                for method, scores in method_scores.items():
                    if chunk_sig in scores:
                        info = scores[chunk_sig]
                        print(f"  {info['title'][:50]}... ({info['date']})")
                        break
    
    return overlap_matrix, chunk_frequency

def analyze_score_distributions(results_dict, case_name="Case"):
    """Compare vector vs LLM scoring patterns"""
    
    print(f"\nSCORE DISTRIBUTION ANALYSIS: {case_name}")
    print("-" * 40)
    
    for method_name, results in results_dict.items():
        normalized_chunks = normalize_chunk_data(results)
        
        vector_scores = [chunk['vector_score'] for chunk in normalized_chunks if chunk['vector_score'] > 0]
        llm_scores = [chunk['llm_score'] for chunk in normalized_chunks if chunk['llm_score'] > 0]
        
        print(f"\n{method_name}:")
        if vector_scores:
            print(f"  Vector scores: μ={np.mean(vector_scores):.3f}, σ={np.std(vector_scores):.3f}, range=[{min(vector_scores):.3f}, {max(vector_scores):.3f}]")
        if llm_scores:
            print(f"  LLM scores: μ={np.mean(llm_scores):.3f}, σ={np.std(llm_scores):.3f}, range=[{min(llm_scores):.3f}, {max(llm_scores):.3f}]")
        
        # Correlation analysis if both scores available
        if vector_scores and llm_scores and len(vector_scores) == len(llm_scores):
            correlation = np.corrcoef(vector_scores, llm_scores)[0,1]
            print(f"  Vector-LLM correlation: {correlation:.3f}")

def analyze_temporal_distribution(results_dict, case_name="Case"):
    """Analyze temporal distribution differences across methods"""
    
    print(f"\nTEMPORAL DISTRIBUTION ANALYSIS: {case_name}")
    print("-" * 40)
    
    periods = {
        '1950-1954': (1950, 1954),
        '1955-1959': (1955, 1959), 
        '1960-1964': (1960, 1964),
        '1965-1969': (1965, 1969),
        '1970-1974': (1970, 1974),
        '1975-1979': (1975, 1979)
    }
    
    method_temporal = {}
    
    for method_name, results in results_dict.items():
        normalized_chunks = normalize_chunk_data(results)
        period_counts = {period: 0 for period in periods.keys()}
        
        for chunk in normalized_chunks:
            date_str = str(chunk.get('date', ''))
            year_match = re.search(r'19\d{2}', date_str)
            
            if year_match:
                year = int(year_match.group())
                for period_name, (start, end) in periods.items():
                    if start <= year <= end:
                        period_counts[period_name] += 1
                        break
        
        method_temporal[method_name] = period_counts
        
        print(f"\n{method_name}:")
        for period, count in period_counts.items():
            print(f"  {period}: {count} chunks")
    
    return method_temporal

def generate_answer_evaluation_template():
    """Generate template for manual answer quality evaluation"""
    
    template = """
# Manual Answer Quality Evaluation - Homosexualität Case

## Evaluation Criteria (1-5 scale each, total /25):

### 1. Historical Accuracy (1-5)
- 5: All facts correct, proper chronology, accurate terminology
- 4: Mostly accurate with minor errors
- 3: Generally accurate with some notable mistakes
- 2: Several significant errors
- 1: Major factual problems

### 2. Temporal Coverage (1-5)
- 5: Excellent coverage of evolution 1950s→1970s, identifies key turning points
- 4: Good temporal progression with most key periods covered
- 3: Adequate coverage but misses some important periods
- 2: Limited temporal scope
- 1: Poor or missing temporal perspective

### 3. Source Integration (1-5)
- 5: Excellent use of sources, clear attribution, balanced perspective
- 4: Good source usage with clear connections
- 3: Adequate source integration
- 2: Limited source utilization
- 1: Poor or minimal source usage

### 4. Analytical Depth (1-5)
- 5: Sophisticated interpretation, identifies patterns and significance
- 4: Good analysis beyond description
- 3: Some analytical insights
- 2: Mostly descriptive with limited analysis
- 1: Purely descriptive, no interpretation

### 5. Methodological Transparency (1-5)
- 5: Clear source traceability, acknowledges limitations, transparent reasoning
- 4: Good transparency with minor gaps
- 3: Adequate transparency
- 2: Limited transparency
- 1: Poor or missing methodological clarity

## Evaluation Template:

### Answer A (Standard Retrieval):
- Historical Accuracy: __/5
- Temporal Coverage: __/5  
- Source Integration: __/5
- Analytical Depth: __/5
- Methodological Transparency: __/5
**Total: __/25**

Notes: 

### Answer B (OpenAI LLM-Assisted):
- Historical Accuracy: __/5
- Temporal Coverage: __/5  
- Source Integration: __/5
- Analytical Depth: __/5
- Methodological Transparency: __/5
**Total: __/25**

Notes:

### Answer C (Gemini LLM-Assisted):
- Historical Accuracy: __/5
- Temporal Coverage: __/5  
- Source Integration: __/5
- Analytical Depth: __/5
- Methodological Transparency: __/5
**Total: __/25**

Notes:

### Overall Ranking:
1st: _______ (Score: __/25)
2nd: _______ (Score: __/25)  
3rd: _______ (Score: __/25)

### Key Differences Observed:
- Best historical accuracy: 
- Best temporal coverage:
- Best source integration:
- Most analytical depth:
- Most transparent methodology:
    """
    
    with open('answer_evaluation_template.txt', 'w', encoding='utf-8') as f:
        f.write(template)
    
    print("Generated answer_evaluation_template.txt")
    print("Complete this evaluation after generating your three answers")

def main():
    """Run the redesigned evaluation"""
    
    print("=== REDESIGNED SPIEGEL RAG EVALUATION ===")
    print("Entnazifizierung: Retrieval Analysis")
    print("Homosexualität: Answer Quality Assessment")
    
    # ENTNAZIFIZIERUNG RETRIEVAL ANALYSIS
    print(f"\n{'='*80}")
    print("CASE A: ENTNAZIFIZIERUNG - RETRIEVAL MECHANICS")
    print(f"{'='*80}")
    
    entnazi_files = {
        'standard_full': 'texte/entnaziG_standard_full.json',
        'standard_temporal': 'texte/entnaziG_standard_temporal.json',
        'openai_temporal': 'texte/entnaziG_openai_temporal.json',
        'gemini_temporal': 'texte/entnaziG_gemini_temporal.json',
        'openai_negative': 'texte/entnaziG_openai_negative.json'
    }
    
    # Load all results
    entnazi_results = {}
    for method, filepath in entnazi_files.items():
        entnazi_results[method] = load_results(filepath)
    
    # Run retrieval analysis
    if any(r['chunks'] for r in entnazi_results.values()):
        overlap_matrix, chunk_freq = analyze_retrieval_overlap(entnazi_results, "Entnazifizierung")
        analyze_score_distributions(entnazi_results, "Entnazifizierung")
        temporal_dist = analyze_temporal_distribution(entnazi_results, "Entnazifizierung")
        
        # Consistency test if available
        consistency_files = {
            'run1': 'entnaziG_openai_consistency1.json',
            'run2': 'entnaziG_openai_consistency2.json'
        }
        consistency_results = {}
        for run, filepath in consistency_files.items():
            consistency_results[run] = load_results(filepath)
        
        if all(r['chunks'] for r in consistency_results.values()):
            print(f"\nCONSISTENCY TEST (OpenAI at temp=0):")
            overlap_matrix_consistency, _ = analyze_retrieval_overlap(consistency_results, "Consistency")
    
    # HOMOSEXUALITÄT LLM vs VECTOR SCORING ANALYSIS
    print(f"\n{'='*80}")
    print("CASE B: HOMOSEXUALITÄT - LLM vs VECTOR SCORING")
    print(f"{'='*80}")
    
    homosex_file = 'texte/homosex_gemini_zeitintervall.json'
    homosex_results = load_results(homosex_file)
    
    if homosex_results.get('chunks'):
        print(f"Loaded {len(homosex_results['chunks'])} chunks for scoring analysis")
        
        # Generate manual scoring template for 30 random chunks
        print("Generating manual scoring template for LLM vs Vector comparison...")
        generate_manual_scoring_template_homosex(homosex_results)
        
        # Analyze scoring patterns
        analyze_llm_vs_vector_scoring(homosex_results)
        
        print("\nDemo preparation:")
        print("Meta-question: 'Wie veränderte sich die Sprache und Argumentation in der")
        print("Berichterstattung über Homosexualität zwischen den 1950er und 1970er Jahren,")
        print("und was verrät dies über gesellschaftliche Wandlungsprozesse?'")
        
        # Check for good demo content
        check_demo_content_quality(homosex_results)
    else:
        print("Homosexualität results not found - run retrieval first")

def generate_manual_scoring_template_homosex(results):
    """Generate CSV template for manual LLM vs Vector scoring comparison"""
    import csv
    import random
    
    normalized_chunks = normalize_chunk_data(results)
    # Random sample of 30 chunks
    sample = random.sample(normalized_chunks, min(30, len(normalized_chunks)))
    
    scoring_data = []
    for i, chunk in enumerate(sample):
        scoring_data.append({
            'chunk_id': f"chunk_{i+1}",
            'title': chunk.get('title', 'No title'),
            'date': chunk.get('date', 'No date'),
            'content_preview': chunk.get('content', '')[:300] + '...',
            'vector_score': chunk.get('vector_score', 0),
            'llm_score': chunk.get('llm_score', 0),
            'manual_score': '',  # To be filled: 0-3 scale
            'notes': ''
        })
    
    # Shuffle for blind evaluation
    random.shuffle(scoring_data)
    
    # Write to CSV
    with open('homosex_manual_scoring.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['chunk_id', 'title', 'date', 'content_preview', 
                                              'vector_score', 'llm_score', 'manual_score', 'notes'])
        writer.writeheader()
        writer.writerows(scoring_data)
    
    print(f"Generated homosex_manual_scoring.csv with {len(scoring_data)} chunks")
    print("Score each chunk 0-3 for relevance to homosexuality query")

def analyze_llm_vs_vector_scoring(results):
    """Analyze correlation between LLM and vector scoring"""
    normalized_chunks = normalize_chunk_data(results)
    
    vector_scores = [chunk['vector_score'] for chunk in normalized_chunks if chunk['vector_score'] > 0]
    llm_scores = [chunk['llm_score'] for chunk in normalized_chunks if chunk['llm_score'] > 0]
    
    if vector_scores and llm_scores and len(vector_scores) == len(llm_scores):
        correlation = np.corrcoef(vector_scores, llm_scores)[0,1]
        
        print(f"\nSCORING ANALYSIS:")
        print(f"Vector scores: μ={np.mean(vector_scores):.3f}, σ={np.std(vector_scores):.3f}")
        print(f"LLM scores: μ={np.mean(llm_scores):.3f}, σ={np.std(llm_scores):.3f}")
        print(f"Correlation: r={correlation:.3f}")
        
        # Top chunks by each method
        print(f"\nTOP CHUNKS BY VECTOR SIMILARITY:")
        sorted_by_vector = sorted(normalized_chunks, key=lambda x: x.get('vector_score', 0), reverse=True)
        for i, chunk in enumerate(sorted_by_vector[:3]):
            title = chunk.get('title', 'No title')[:50]
            v_score = chunk.get('vector_score', 0)
            l_score = chunk.get('llm_score', 0)
            print(f"  {i+1}. {title}... (V:{v_score:.3f}, L:{l_score:.1f})")
        
        print(f"\nTOP CHUNKS BY LLM EVALUATION:")
        sorted_by_llm = sorted(normalized_chunks, key=lambda x: x.get('llm_score', 0), reverse=True)
        for i, chunk in enumerate(sorted_by_llm[:3]):
            title = chunk.get('title', 'No title')[:50]
            v_score = chunk.get('vector_score', 0)
            l_score = chunk.get('llm_score', 0)
            print(f"  {i+1}. {title}... (V:{v_score:.3f}, L:{l_score:.1f})")

def check_demo_content_quality(results):
    """Check for good content for meta-question demo"""
    normalized_chunks = normalize_chunk_data(results)
    
    # Check temporal distribution
    periods = {'1950s': 0, '1960s': 0, '1970s': 0}
    key_terms = {'paragraph 175': 0, 'bekennt': 0, 'schwul': 0, 'bewegung': 0}
    
    for chunk in normalized_chunks:
        content = chunk.get('content', '').lower()
        date_str = str(chunk.get('date', ''))
        
        # Check periods
        if '195' in date_str:
            periods['1950s'] += 1
        elif '196' in date_str:
            periods['1960s'] += 1
        elif '197' in date_str:
            periods['1970s'] += 1
            
        # Check key terms
        for term in key_terms:
            if term in content:
                key_terms[term] += 1
    
    print(f"\nDEMO CONTENT QUALITY CHECK:")
    print(f"Temporal distribution: {periods}")
    print(f"Key terms found: {key_terms}")
    
    # Check for high-quality analytical chunks
    high_llm_score_chunks = [c for c in normalized_chunks if c.get('llm_score', 0) >= 0.8]
    print(f"High LLM score chunks (≥0.8): {len(high_llm_score_chunks)}")

def analyze_manual_vs_automated_scoring():
    """Compare manual relevance scores with automated scoring"""
    try:
        df = pd.read_csv('homosex_manual_scoring.csv')
        
        print("\n=== MANUAL vs AUTOMATED SCORING COMPARISON ===")
        
        # Remove rows where manual_score is empty
        df = df[df['manual_score'] != '']
        df['manual_score'] = pd.to_numeric(df['manual_score'])
        
        # Correlations
        manual_vector_corr = df['manual_score'].corr(df['vector_score'])
        manual_llm_corr = df['manual_score'].corr(df['llm_score'])
        vector_llm_corr = df['vector_score'].corr(df['llm_score'])
        
        print(f"Manual vs Vector correlation: r={manual_vector_corr:.3f}")
        print(f"Manual vs LLM correlation: r={manual_llm_corr:.3f}")
        print(f"Vector vs LLM correlation: r={vector_llm_corr:.3f}")
        
        # Score distributions
        print(f"\nScore distributions:")
        print(f"Manual scores: μ={df['manual_score'].mean():.2f}, σ={df['manual_score'].std():.2f}")
        print(f"Vector scores: μ={df['vector_score'].mean():.3f}, σ={df['vector_score'].std():.3f}")
        print(f"LLM scores: μ={df['llm_score'].mean():.2f}, σ={df['llm_score'].std():.2f}")
        
        # Which method better predicts manual scores?
        if manual_llm_corr > manual_vector_corr:
            print(f"\n✓ LLM scoring better predicts manual relevance (r={manual_llm_corr:.3f} vs r={manual_vector_corr:.3f})")
        else:
            print(f"\n✓ Vector scoring better predicts manual relevance (r={manual_vector_corr:.3f} vs r={manual_llm_corr:.3f})")
            
        return manual_vector_corr, manual_llm_corr, vector_llm_corr
        
    except FileNotFoundError:
        print("homosex_manual_scoring.csv not found - complete manual scoring first")
        return None, None, None
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print("1. Complete Entnazifizierung retrieval runs (7 JSON files)")
    print("2. Generate Homosexualität answers (3 text files)")
    print("3. Complete manual answer evaluation using template")
    print("4. Create presentation visualizations from results")

def create_presentation_visualizations():
    """Generate charts for presentation"""
    # This function would create the overlap matrices, score distributions, etc.
    # Can be called after data collection is complete
    pass

if __name__ == "__main__":
    main()