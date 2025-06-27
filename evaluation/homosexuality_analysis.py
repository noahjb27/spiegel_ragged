#!/usr/bin/env python3
"""
Test script for the fixed create_homosex_scoring_analysis function
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set up matplotlib for high-quality output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

# Create output directory
os.makedirs('visualizations', exist_ok=True)

def load_results(filepath):
    """Load JSON results from search"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✓ Loaded {filepath}")
            return data
    except Exception as e:
        print(f"✗ Error loading {filepath}: {e}")
        return {'chunks': []}

def normalize_chunk_data(results):
    """Normalize chunk data from different result formats"""
    normalized_chunks = []
    
    if not isinstance(results, dict):
        return normalized_chunks
    
    # Handle LLM-assisted results (evaluations array)
    heuristik_metadata = results.get('heuristik_metadata', {})
    if 'evaluations' in heuristik_metadata:
        evaluations = heuristik_metadata.get('evaluations', [])
        
        for eval_item in evaluations:
            original_llm_score = eval_item.get('original_llm_score', 0)
            llm_score = eval_item.get('llm_score', 0)
            
            if original_llm_score > 0:
                display_llm_score = float(original_llm_score)
            elif llm_score > 0 and llm_score <= 1.0:
                display_llm_score = float(llm_score * 10)
            else:
                display_llm_score = float(llm_score)
                
            normalized_chunks.append({
                'title': str(eval_item.get('title', 'No title')),
                'date': str(eval_item.get('date', 'No date')),
                'vector_score': float(eval_item.get('vector_score', 0)),
                'llm_score': display_llm_score,
                'content': str(eval_item.get('evaluation', ''))[:200],
                'source_type': 'llm_assisted'
            })
    
    # Handle standard search results (chunks array)
    elif 'chunks' in results:
        chunks = results.get('chunks', [])
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            normalized_chunks.append({
                'title': str(metadata.get('titel', 'No title')),
                'date': str(metadata.get('datum', 'No date')),
                'vector_score': float(chunk.get('relevance_score', 0)),
                'llm_score': 0.0,
                'content': str(chunk.get('content', ''))[:200],
                'source_type': 'standard'
            })
    
    return normalized_chunks

def get_chunk_signature(chunk):
    """Create unique identifier for chunk comparison"""
    title = str(chunk.get('title', 'unknown')).strip()
    date = str(chunk.get('date', 'unknown')).strip()
    title_clean = title.replace('»', '"').replace('«', '"').lower()
    return f"{title_clean}_{date}"

# Insert the fixed function here
def create_homosex_scoring_analysis(results, case_name="Homosexuality"):
    """Create specific visualizations for homosexuality scoring analysis including manual scores"""
    normalized_chunks = normalize_chunk_data(results)
    
    # Try to load manual scores from CSV
    manual_scores_data = {}
    has_manual_scores = False
    manual_score_count = ho0
    
    try:
        df_manual = pd.read_csv('homosex_manual_scoring.csv')
        print(f"📊 Loaded manual scoring CSV with {len(df_manual)} rows")
        
        # Create lookup by title and date with better matching
        for _, row in df_manual.iterrows():
            if not pd.isna(row['manual_score']) and row['manual_score'] != '':
                # Use the same key generation as get_chunk_signature for consistency
                title = str(row['title']).strip().replace('»', '"').replace('«', '"').lower()
                date = str(row['date']).strip()
                key = f"{title}_{date}"
                manual_scores_data[key] = float(row['manual_score'])
                manual_score_count += 1
        
        if manual_scores_data:
            has_manual_scores = True
            print(f"✅ Found {len(manual_scores_data)} manual scores to include in visualization")
        else:
            print("⚠️ No manual scores found in CSV - using automated scores only")
            
    except Exception as e:
        print(f"⚠️ Could not load manual scores: {e} - using automated scores only")
    
    # Create figure with proper layout
    if has_manual_scores:
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))  # Changed to 3x2 for better layout
        fig.suptitle(f'{case_name}: Scoring Analysis (with Manual Scores)', 
                     fontsize=16, fontweight='bold', y=0.98)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{case_name}: Scoring Analysis', 
                     fontsize=16, fontweight='bold', y=0.98)
    
    # Collect scores and match with manual scores
    vector_scores = []
    llm_scores = []
    manual_scores = []
    matched_chunks = []
    
    print(f"Processing {len(normalized_chunks)} normalized chunks...")
    
    for chunk in normalized_chunks:
        if chunk['vector_score'] > 0:
            vector_scores.append(chunk['vector_score'])
            llm_scores.append(chunk['llm_score'])
            
            # Check for manual score using same signature generation
            if has_manual_scores:
                chunk_sig = get_chunk_signature(chunk)
                manual_score = manual_scores_data.get(chunk_sig, None)
                manual_scores.append(manual_score if manual_score is not None else np.nan)
                
                if manual_score is not None:
                    matched_chunks.append({
                        'vector': chunk['vector_score'],
                        'llm': chunk['llm_score'], 
                        'manual': manual_score,
                        'title': chunk['title'],
                        'date': chunk['date']
                    })
            else:
                manual_scores.append(np.nan)
    
    print(f"Collected {len(vector_scores)} vector scores, {len(llm_scores)} LLM scores")
    if has_manual_scores:
        valid_manual_count = sum(1 for score in manual_scores if not np.isnan(score))
        print(f"Matched {valid_manual_count}/{len(manual_scores)} chunks with manual scores")
    
    # 1. Vector vs LLM correlation
    ax1 = axes[0,0]
    if vector_scores and llm_scores and len(vector_scores) == len(llm_scores):
        correlation = np.corrcoef(vector_scores, llm_scores)[0,1]
        
        ax1.scatter(vector_scores, llm_scores, alpha=0.6, s=50, color='darkblue', edgecolors='white', linewidth=0.5)
        ax1.set_xlabel('Vector Score', fontweight='bold')
        ax1.set_ylabel('LLM Score', fontweight='bold')
        ax1.set_title(f'Vector vs LLM Correlation (r={correlation:.3f})', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Add trend line
        if len(vector_scores) > 1:
            z = np.polyfit(vector_scores, llm_scores, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(vector_scores), max(vector_scores), 100)
            ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    # 2. Score distribution comparison
    ax2 = axes[0,1]
    
    score_data = []
    if vector_scores:
        score_data.extend([('Vector', score) for score in vector_scores])
    if llm_scores:
        score_data.extend([('LLM', score) for score in llm_scores])
    if has_manual_scores and manual_scores:
        valid_manual = [score for score in manual_scores if not np.isnan(score)]
        if valid_manual:
            score_data.extend([('Manual', score) for score in valid_manual])
    
    if score_data:
        df_scores = pd.DataFrame(score_data, columns=['Type', 'Score'])
        score_types = df_scores['Type'].unique()
        colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(score_types)]
        
        for i, score_type in enumerate(score_types):
            data = df_scores[df_scores['Type'] == score_type]['Score']
            ax2.hist(data, bins=15, alpha=0.7, color=colors[i], 
                    label=f'{score_type} (μ={np.mean(data):.2f})', edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Score', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Score Distributions Comparison', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    # 3. Temporal distribution
    ax3 = axes[1,0]
    periods = {'1950s': 0, '1960s': 0, '1970s': 0}
    for chunk in normalized_chunks:
        date_str = str(chunk.get('date', ''))
        if '195' in date_str:
            periods['1950s'] += 1
        elif '196' in date_str:
            periods['1960s'] += 1
        elif '197' in date_str:
            periods['1970s'] += 1
    
    periods_list = list(periods.keys())
    counts = list(periods.values())
    colors_temp = ['#FF9999', '#66B2FF', '#99FF99']
    
    bars = ax3.bar(periods_list, counts, color=colors_temp, alpha=0.8, edgecolor='black', linewidth=1)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xlabel('Time Period', fontweight='bold')
    ax3.set_ylabel('Number of Chunks', fontweight='bold')
    ax3.set_title('Temporal Distribution', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Statistics summary
    ax4 = axes[1,1]
    stats_text = []
    
    if vector_scores:
        stats_text.append(f"Vector: μ={np.mean(vector_scores):.3f}, σ={np.std(vector_scores):.3f}")
        stats_text.append(f"        range=[{min(vector_scores):.3f}, {max(vector_scores):.3f}]")
    if llm_scores:
        stats_text.append(f"LLM:    μ={np.mean(llm_scores):.2f}, σ={np.std(llm_scores):.2f}")
        stats_text.append(f"        range=[{min(llm_scores):.1f}, {max(llm_scores):.1f}]")
    
    # Manual score analysis with correlations
    if has_manual_scores and matched_chunks:
        manual_values = [chunk['manual'] for chunk in matched_chunks]
        vector_values = [chunk['vector'] for chunk in matched_chunks]
        llm_values = [chunk['llm'] for chunk in matched_chunks]
        
        stats_text.append(f"Manual: μ={np.mean(manual_values):.2f}, σ={np.std(manual_values):.2f}")
        stats_text.append(f"        range=[{min(manual_values):.0f}, {max(manual_values):.0f}]")
        stats_text.append(f"        n={len(manual_values)} scored chunks")
        
        # Calculate correlations
        if len(manual_values) > 1:
            try:
                corr_mv = np.corrcoef(manual_values, vector_values)[0,1]
                corr_ml = np.corrcoef(manual_values, llm_values)[0,1]
                
                stats_text.append("")
                stats_text.append("Correlations:")
                stats_text.append(f"Manual-Vector: r={corr_mv:.3f}")
                stats_text.append(f"Manual-LLM:    r={corr_ml:.3f}")
                
                # Determine which automated method better predicts manual scores
                if abs(corr_ml) > abs(corr_mv):
                    stats_text.append("→ LLM better predicts manual")
                elif abs(corr_mv) > abs(corr_ml):
                    stats_text.append("→ Vector better predicts manual")
                else:
                    stats_text.append("→ Similar predictive power")
                    
            except Exception as e:
                stats_text.append(f"Correlation calc error: {e}")
    
    ax4.text(0.05, 0.95, '\n'.join(stats_text), transform=ax4.transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    ax4.set_title('Score Statistics', fontweight='bold')
    ax4.axis('off')
    
    # 5 & 6. Manual vs Automated correlations (only if manual scores available)
    if has_manual_scores and matched_chunks and len(matched_chunks) > 1:
        manual_values = [chunk['manual'] for chunk in matched_chunks]
        vector_values = [chunk['vector'] for chunk in matched_chunks]
        llm_values = [chunk['llm'] for chunk in matched_chunks]
        
        try:
            # Manual vs Vector
            ax5 = axes[2,0]
            corr_mv = np.corrcoef(manual_values, vector_values)[0,1]
            ax5.scatter(vector_values, manual_values, alpha=0.7, s=60, color='green', edgecolors='white', linewidth=0.5)
            ax5.set_xlabel('Vector Score', fontweight='bold')
            ax5.set_ylabel('Manual Score', fontweight='bold')
            ax5.set_title(f'Manual vs Vector (r={corr_mv:.3f})', fontweight='bold')
            ax5.grid(alpha=0.3)
            ax5.set_ylim(-0.2, 3.2)  # Manual scores are 0-3
            
            # Add trend line
            z = np.polyfit(vector_values, manual_values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(vector_values), max(vector_values), 100)
            ax5.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            # Manual vs LLM  
            ax6 = axes[2,1]
            corr_ml = np.corrcoef(manual_values, llm_values)[0,1]
            ax6.scatter(llm_values, manual_values, alpha=0.7, s=60, color='purple', edgecolors='white', linewidth=0.5)
            ax6.set_xlabel('LLM Score', fontweight='bold')
            ax6.set_ylabel('Manual Score', fontweight='bold')
            ax6.set_title(f'Manual vs LLM (r={corr_ml:.3f})', fontweight='bold')
            ax6.grid(alpha=0.3)
            ax6.set_ylim(-0.2, 3.2)  # Manual scores are 0-3
            
            # Add trend line
            z = np.polyfit(llm_values, manual_values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(llm_values), max(llm_values), 100)
            ax6.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
        except Exception as e:
            print(f"Error creating correlation plots: {e}")
            # Hide the axes if correlation plots fail
            axes[2,0].axis('off')
            axes[2,1].axis('off')
    
    elif has_manual_scores:
        # Hide unused axes if no sufficient manual data
        axes[2,0].axis('off')
        axes[2,1].axis('off')
    
    plt.tight_layout()
    filename = f'visualizations/{case_name.lower()}_scoring_analysis_fixed.jpg'
    plt.savefig(filename, format='jpg', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Created {filename} {'with manual scores integrated' if has_manual_scores else 'with automated scores only'}")
    return filename

def main():
    """Test the fixed function"""
    print("Testing fixed create_homosex_scoring_analysis function")
    
    # Load the homosexuality results
    homosex_file = 'texte/homosex_openai_zeitintervall.json'
    homosex_results = load_results(homosex_file)
    
    if homosex_results.get('chunks') or homosex_results.get('heuristik_metadata', {}).get('evaluations'):
        print(f"\n🧪 Testing fixed visualization function...")
        filename = create_homosex_scoring_analysis(homosex_results, "Homosexuality")
        print(f"\n✅ Fixed visualization created: {filename}")
        
        # Also create a simple correlation analysis
        print(f"\n📊 Quick correlation analysis:")
        normalized_chunks = normalize_chunk_data(homosex_results)
        
        # Load manual scores for correlation
        try:
            df_manual = pd.read_csv('homosex_manual_scoring.csv')
            print(f"Manual scores loaded: {len(df_manual)} entries")
            
            # Get correlation with manual scores
            manual_data = []
            for _, row in df_manual.iterrows():
                if not pd.isna(row['manual_score']) and row['manual_score'] != '':
                    title_clean = str(row['title']).strip().replace('»', '"').replace('«', '"').lower()
                    date_clean = str(row['date']).strip()
                    key = f"{title_clean}_{date_clean}"
                    
                    # Find matching chunk
                    for chunk in normalized_chunks:
                        chunk_sig = get_chunk_signature(chunk)
                        if chunk_sig == key:
                            manual_data.append({
                                'manual': float(row['manual_score']),
                                'vector': chunk['vector_score'],
                                'llm': chunk['llm_score'],
                                'title': chunk['title']
                            })
                            break
            
            if manual_data:
                manual_scores = [d['manual'] for d in manual_data]
                vector_scores = [d['vector'] for d in manual_data]
                llm_scores = [d['llm'] for d in manual_data]
                
                corr_mv = np.corrcoef(manual_scores, vector_scores)[0,1]
                corr_ml = np.corrcoef(manual_scores, llm_scores)[0,1]
                
                print(f"Manual vs Vector correlation: r={corr_mv:.3f}")
                print(f"Manual vs LLM correlation: r={corr_ml:.3f}")
                
                if abs(corr_ml) > abs(corr_mv):
                    print("→ LLM scores better predict manual relevance judgments")
                else:
                    print("→ Vector scores better predict manual relevance judgments")
                    
                print(f"Matched {len(manual_data)} chunks for correlation analysis")
            else:
                print("❌ No matching chunks found for correlation analysis")
                
        except Exception as e:
            print(f"❌ Error in correlation analysis: {e}")
    else:
        print("❌ No homosexuality results found")

if __name__ == "__main__":
    main()