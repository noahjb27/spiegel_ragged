#!/usr/bin/env python3
"""
Visualization script for SPIEGEL RAG evaluation results
Generates charts for presentation slides
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re

# Set style for presentation-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def load_results(filepath):
    """Load JSON results from search"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
        return None

def normalize_chunk_data(results):
    """Normalize chunk data from different result formats"""
    if not results:
        return []
        
    normalized_chunks = []
    
    # Handle LLM-assisted results (evaluations array)
    heuristik_metadata = results.get('heuristik_metadata', {})
    if 'evaluations' in heuristik_metadata:
        for eval_item in heuristik_metadata['evaluations']:
            llm_score = eval_item.get('original_llm_score', eval_item.get('llm_score', 0))
            if isinstance(llm_score, (int, float)) and llm_score <= 1.0:
                llm_score = llm_score * 10
                
            normalized_chunks.append({
                'title': str(eval_item.get('title', 'No title')),
                'date': str(eval_item.get('date', 'No date')),
                'vector_score': float(eval_item.get('vector_score', 0)),
                'llm_score': float(llm_score),
                'window': str(eval_item.get('window', '')),
                'source_type': 'llm_assisted'
            })
    
    # Handle standard search results (chunks array)
    elif 'chunks' in results:
        for chunk in results['chunks']:
            metadata = chunk.get('metadata', {})
            normalized_chunks.append({
                'title': str(metadata.get('titel', 'No title')),
                'date': str(metadata.get('datum', 'No date')),
                'vector_score': float(chunk.get('relevance_score', 0)),
                'llm_score': 0.0,
                'window': '',
                'source_type': 'standard'
            })
    
    return normalized_chunks

def create_overlap_matrix(results_dict, title="Method Overlap Matrix", save_path="overlap_matrix.png"):
    """Create heatmap showing overlap percentages between methods"""
    
    methods = list(results_dict.keys())
    n_methods = len(methods)
    overlap_matrix = np.zeros((n_methods, n_methods))
    
    # Calculate overlaps
    method_chunks = {}
    for i, method in enumerate(methods):
        chunks = normalize_chunk_data(results_dict[method])
        signatures = [f"{chunk['title']}_{chunk['date']}" for chunk in chunks]
        method_chunks[method] = set(signatures)
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                overlap_matrix[i, j] = 100.0  # Self-overlap is 100%
            else:
                overlap = len(method_chunks[method1] & method_chunks[method2])
                total = len(method_chunks[method1])
                overlap_pct = (overlap / total * 100) if total > 0 else 0
                overlap_matrix[i, j] = overlap_pct
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    # Clean method names for display
    display_methods = [method.replace('_', ' ').title() for method in methods]
    
    sns.heatmap(overlap_matrix, 
                annot=True, 
                fmt='.1f',
                xticklabels=display_methods,
                yticklabels=display_methods,
                cmap='RdYlBu_r',
                center=50,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Overlap Percentage (%)'})
    
    plt.title(f'{title}\n(% of chunks shared between methods)', fontsize=14, pad=20)
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Method', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved overlap matrix to {save_path}")

def create_score_distribution_plot(results_dict, title="Score Distributions", save_path="score_distributions.png"):
    """Create box plot comparing vector vs LLM score distributions"""
    
    score_data = []
    
    for method_name, results in results_dict.items():
        chunks = normalize_chunk_data(results)
        
        for chunk in chunks:
            # Vector scores
            if chunk['vector_score'] > 0:
                score_data.append({
                    'Method': method_name.replace('_', ' ').title(),
                    'Score Type': 'Vector Similarity',
                    'Score': chunk['vector_score']
                })
            
            # LLM scores
            if chunk['llm_score'] > 0:
                score_data.append({
                    'Method': method_name.replace('_', ' ').title(),
                    'Score Type': 'LLM Evaluation',
                    'Score': chunk['llm_score'] / 10.0  # Normalize to 0-1 for comparison
                })
    
    if not score_data:
        print("No score data found for visualization")
        return
    
    df = pd.DataFrame(score_data)
    
    plt.figure(figsize=(12, 8))
    
    sns.boxplot(data=df, x='Method', y='Score', hue='Score Type')
    
    plt.title(f'{title}\n(Vector Similarity vs LLM Evaluation)', fontsize=14, pad=20)
    plt.xlabel('Retrieval Method', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Score Type', title_fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved score distribution plot to {save_path}")

def create_temporal_distribution_plot(results_dict, title="Temporal Distribution", save_path="temporal_distribution.png"):
    """Create stacked bar chart showing temporal distribution by method"""
    
    periods = ['1950-1954', '1955-1959', '1960-1964', '1965-1969', '1970-1974', '1975-1979']
    
    temporal_data = {}
    
    for method_name, results in results_dict.items():
        chunks = normalize_chunk_data(results)
        period_counts = {period: 0 for period in periods}
        
        for chunk in chunks:
            date_str = str(chunk.get('date', ''))
            year_match = re.search(r'19\d{2}', date_str)
            
            if year_match:
                year = int(year_match.group())
                for period in periods:
                    start, end = map(int, period.split('-'))
                    if start <= year <= end:
                        period_counts[period] += 1
                        break
        
        temporal_data[method_name] = period_counts
    
    # Create DataFrame for plotting
    df_data = []
    for method, counts in temporal_data.items():
        for period, count in counts.items():
            df_data.append({
                'Method': method.replace('_', ' ').title(),
                'Period': period,
                'Count': count
            })
    
    df = pd.DataFrame(df_data)
    
    plt.figure(figsize=(14, 8))
    
    # Create stacked bar chart
    pivot_df = df.pivot(index='Method', columns='Period', values='Count').fillna(0)
    
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(14, 8), 
                       colormap='viridis', alpha=0.8)
    
    plt.title(f'{title}\n(Number of chunks per time period)', fontsize=14, pad=20)
    plt.xlabel('Retrieval Method', fontsize=12)
    plt.ylabel('Number of Chunks', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Time Period', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved temporal distribution plot to {save_path}")

def create_consistency_analysis(consistency_files, title="Consistency Analysis", save_path="consistency_analysis.png"):
    """Create visualization for LLM consistency test"""
    
    if len(consistency_files) != 2:
        print("Consistency analysis requires exactly 2 files")
        return
    
    results = {}
    for name, filepath in consistency_files.items():
        results[name] = load_results(filepath)
    
    # Calculate overlap
    chunks_1 = normalize_chunk_data(results[list(consistency_files.keys())[0]])
    chunks_2 = normalize_chunk_data(results[list(consistency_files.keys())[1]])
    
    sigs_1 = set(f"{chunk['title']}_{chunk['date']}" for chunk in chunks_1)
    sigs_2 = set(f"{chunk['title']}_{chunk['date']}" for chunk in chunks_2)
    
    overlap = len(sigs_1 & sigs_2)
    unique_1 = len(sigs_1 - sigs_2)
    unique_2 = len(sigs_2 - sigs_1)
    
    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart of overlap
    labels = ['Overlapping Chunks', 'Run 1 Unique', 'Run 2 Unique']
    sizes = [overlap, unique_1, unique_2]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Consistency Between Identical Runs\n(OpenAI at Temperature 0)', fontsize=12)
    
    # Score correlation if available
    if all(chunk['llm_score'] > 0 for chunk in chunks_1) and all(chunk['llm_score'] > 0 for chunk in chunks_2):
        # Find matching chunks and compare scores
        matching_scores_1 = []
        matching_scores_2 = []
        
        for chunk1 in chunks_1:
            sig1 = f"{chunk1['title']}_{chunk1['date']}"
            for chunk2 in chunks_2:
                sig2 = f"{chunk2['title']}_{chunk2['date']}"
                if sig1 == sig2:
                    matching_scores_1.append(chunk1['llm_score'])
                    matching_scores_2.append(chunk2['llm_score'])
                    break
        
        if matching_scores_1 and matching_scores_2:
            ax2.scatter(matching_scores_1, matching_scores_2, alpha=0.6)
            ax2.plot([0, 10], [0, 10], 'r--', alpha=0.8, label='Perfect Consistency')
            ax2.set_xlabel('Run 1 LLM Scores', fontsize=10)
            ax2.set_ylabel('Run 2 LLM Scores', fontsize=10)
            ax2.set_title('Score Consistency for Overlapping Chunks', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Calculate correlation
            if len(matching_scores_1) > 1:
                correlation = np.corrcoef(matching_scores_1, matching_scores_2)[0,1]
                ax2.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax2.transAxes, 
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved consistency analysis to {save_path}")

def create_presentation_summary_chart(homosex_manual_scores_file="homosex_manual_scoring.csv", 
                                    save_path="homosex_correlation_summary.png"):
    """Create summary chart for homosexuality manual scoring results"""
    
    try:
        df = pd.read_csv(homosex_manual_scores_file)
        
        # Remove empty manual scores
        df = df[df['manual_score'] != ''].copy()
        df['manual_score'] = pd.to_numeric(df['manual_score'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Correlation scatter plots
        ax1.scatter(df['manual_score'], df['vector_score'], alpha=0.7, color='blue', label='Manual vs Vector')
        manual_vector_corr = df['manual_score'].corr(df['vector_score'])
        ax1.set_xlabel('Manual Relevance Score (0-3)')
        ax1.set_ylabel('Vector Similarity Score')
        ax1.set_title(f'Manual vs Vector Scoring\nr = {manual_vector_corr:.3f}')
        ax1.grid(True, alpha=0.3)
        
        ax2.scatter(df['manual_score'], df['llm_score'], alpha=0.7, color='red', label='Manual vs LLM')
        manual_llm_corr = df['manual_score'].corr(df['llm_score'])
        ax2.set_xlabel('Manual Relevance Score (0-3)')
        ax2.set_ylabel('LLM Evaluation Score')
        ax2.set_title(f'Manual vs LLM Scoring\nr = {manual_llm_corr:.3f}')
        ax2.grid(True, alpha=0.3)
        
        # 3. Score distributions
        scores_data = []
        for _, row in df.iterrows():
            scores_data.append({'Score Type': 'Manual', 'Score': row['manual_score']})
            scores_data.append({'Score Type': 'Vector', 'Score': row['vector_score']})
            scores_data.append({'Score Type': 'LLM', 'Score': row['llm_score']/10*3})  # Normalize to 0-3 scale
        
        scores_df = pd.DataFrame(scores_data)
        sns.boxplot(data=scores_df, x='Score Type', y='Score', ax=ax3)
        ax3.set_title('Score Distribution Comparison')
        ax3.set_ylabel('Score (normalized)')
        
        # 4. Summary bar chart
        correlations = {
            'Manual vs Vector': manual_vector_corr,
            'Manual vs LLM': manual_llm_corr,
            'Vector vs LLM': df['vector_score'].corr(df['llm_score'])
        }
        
        bars = ax4.bar(correlations.keys(), correlations.values(), 
                      color=['blue', 'red', 'green'], alpha=0.7)
        ax4.set_ylabel('Correlation Coefficient')
        ax4.set_title('Correlation Summary')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, correlations.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Homosexualität: LLM vs Vector Scoring Validation', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved correlation summary to {save_path}")
        
        return correlations
        
    except FileNotFoundError:
        print(f"Manual scoring file {homosex_manual_scores_file} not found")
        print("Complete manual scoring first, then run this visualization")
        return None

def main():
    """Generate all presentation visualizations"""
    
    print("=== SPIEGEL RAG EVALUATION VISUALIZATIONS ===")
    
    # File paths
    base_path = Path("texte")
    
    entnazi_files = {
        'standard_full': base_path / 'entnaziG_standard_full.json',
        'standard_temporal': base_path / 'entnaziG_standard_temporal.json', 
        'openai_temporal': base_path / 'entnaziG_openai_temporal.json',
        'gemini_temporal': base_path / 'entnaziG_gemini_temporal.json',
        'openai_negative': base_path / 'entnaziG_openai_negative.json'
    }
    
    consistency_files = {
        'run1': base_path / 'entnaziG_openai_consistency1.json',
        'run2': base_path / 'entnaziG_openai_consistency2.json'
    }
    
    # Load Entnazifizierung data
    print("\n1. Creating Entnazifizierung visualizations...")
    entnazi_results = {}
    for name, filepath in entnazi_files.items():
        result = load_results(filepath)
        if result:
            entnazi_results[name] = result
    
    if entnazi_results:
        create_overlap_matrix(entnazi_results, 
                            "Entnazifizierung: Retrieval Method Overlap",
                            "entnazi_overlap_matrix.png")
        
        create_score_distribution_plot(entnazi_results,
                                     "Entnazifizierung: Score Distributions", 
                                     "entnazi_score_distributions.png")
        
        create_temporal_distribution_plot(entnazi_results,
                                        "Entnazifizierung: Temporal Distribution",
                                        "entnazi_temporal_distribution.png")
    
    # Consistency analysis
    print("\n2. Creating consistency analysis...")
    consistency_results = {}
    for name, filepath in consistency_files.items():
        result = load_results(filepath)
        if result:
            consistency_results[name] = result
    
    if len(consistency_results) == 2:
        create_consistency_analysis(consistency_files,
                                  "OpenAI Consistency Test",
                                  "consistency_analysis.png")
    
    # Homosexualität correlation analysis
    print("\n3. Creating Homosexualität scoring analysis...")
    correlations = create_presentation_summary_chart()
    
    if correlations:
        print(f"\nKey findings for presentation:")
        print(f"- Manual vs Vector correlation: {correlations['Manual vs Vector']:.3f}")
        print(f"- Manual vs LLM correlation: {correlations['Manual vs LLM']:.3f}")
        
        if correlations['Manual vs LLM'] > correlations['Manual vs Vector']:
            print("✓ LLM scoring better predicts human relevance judgment")
        else:
            print("✓ Vector scoring better predicts human relevance judgment")
    
    print(f"\n=== Visualization complete! ===")
    print("Generated files:")
    print("- entnazi_overlap_matrix.png")
    print("- entnazi_score_distributions.png") 
    print("- entnazi_temporal_distribution.png")
    print("- consistency_analysis.png")
    print("- homosex_correlation_summary.png")

if __name__ == "__main__":
    main()