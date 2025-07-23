#!/usr/bin/env python3
"""
PPL Distribution Analysis Script
Analyzes and visualizes the perplexity distribution of improvement and regression samples,
sorted by base model PPL to show patterns.
"""

import argparse
import jsonlines
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Add scripts directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import from complete_token_analysis
from complete_token_analysis import TokenAnalyzer, load_and_match_data


def analyze_ppl_distribution(analyzer, base_data, enhanced_data, improvements, regressions, 
                           n_tokens=100, include_problem=True):
    """Analyze PPL distribution for improvement and regression samples."""
    
    results = {
        'improvements': [],
        'regressions': []
    }
    
    # Analyze improvements
    logger.info(f"Analyzing PPL distribution for {len(improvements)} improvement cases...")
    for base_idx, enh_idx in tqdm(improvements, desc="Improvements"):
        problem_text = base_data[base_idx]['doc']['problem']
        base_resp = base_data[base_idx]["resps"][0][0]
        enh_resp = enhanced_data[enh_idx]["resps"][0][0]
        
        if include_problem:
            # Use the new method that analyzes only answer tokens but with problem context
            base_tokens, base_ppls, _ = analyzer.compute_answer_token_metrics(problem_text, base_resp, n_tokens)
            enh_tokens, enh_ppls, _ = analyzer.compute_answer_token_metrics(problem_text, enh_resp, n_tokens)
        else:
            # Analyze only the response text
            base_tokens, base_ppls, _ = analyzer.compute_token_metrics(base_resp, n_tokens)
            enh_tokens, enh_ppls, _ = analyzer.compute_token_metrics(enh_resp, n_tokens)
        
        if base_ppls and enh_ppls:
            results['improvements'].append({
                'base_idx': base_idx,
                'enh_idx': enh_idx,
                'base_avg_ppl': np.mean(base_ppls),
                'enh_avg_ppl': np.mean(enh_ppls),
                'base_ppls': base_ppls,
                'enh_ppls': enh_ppls
            })
    
    # Analyze regressions
    logger.info(f"Analyzing PPL distribution for {len(regressions)} regression cases...")
    for base_idx, enh_idx in tqdm(regressions, desc="Regressions"):
        problem_text = base_data[base_idx]['doc']['problem']
        base_resp = base_data[base_idx]["resps"][0][0]
        enh_resp = enhanced_data[enh_idx]["resps"][0][0]
        
        if include_problem:
            # Use the new method that analyzes only answer tokens but with problem context
            base_tokens, base_ppls, _ = analyzer.compute_answer_token_metrics(problem_text, base_resp, n_tokens)
            enh_tokens, enh_ppls, _ = analyzer.compute_answer_token_metrics(problem_text, enh_resp, n_tokens)
        else:
            # Analyze only the response text
            base_tokens, base_ppls, _ = analyzer.compute_token_metrics(base_resp, n_tokens)
            enh_tokens, enh_ppls, _ = analyzer.compute_token_metrics(enh_resp, n_tokens)
        
        if base_ppls and enh_ppls:
            results['regressions'].append({
                'base_idx': base_idx,
                'enh_idx': enh_idx,
                'base_avg_ppl': np.mean(base_ppls),
                'enh_avg_ppl': np.mean(enh_ppls),
                'base_ppls': base_ppls,
                'enh_ppls': enh_ppls
            })
    
    # Sort by base PPL
    results['improvements'].sort(key=lambda x: x['base_avg_ppl'])
    results['regressions'].sort(key=lambda x: x['base_avg_ppl'])
    
    return results


def create_ppl_distribution_plots(results, output_dir):
    """Create PPL distribution plots for improvements and regressions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Improvement cases
    if results['improvements']:
        n_improvements = len(results['improvements'])
        x_positions = np.arange(n_improvements)
        
        base_ppls = [r['base_avg_ppl'] for r in results['improvements']]
        enh_ppls = [r['enh_avg_ppl'] for r in results['improvements']]
        
        # Create bar plot
        width = 0.4
        bars1 = ax1.bar(x_positions - width/2, base_ppls, width, label='Base Model', 
                        color='lightblue', alpha=0.7)
        bars2 = ax1.bar(x_positions + width/2, enh_ppls, width, label='Enhanced Model', 
                        color='lightgreen', alpha=0.7)
        
        ax1.set_xlabel('Sample Index (sorted by Base PPL)')
        ax1.set_ylabel('Average Perplexity (log scale)')
        ax1.set_yscale('log')
        ax1.set_title(f'Improvement Cases: PPL Distribution (n={n_improvements})', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y', which='both')
        
        # Add a horizontal line at PPL=100 for reference
        ax1.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='PPL=100')
        
        # Limit x-axis ticks for readability
        if n_improvements > 20:
            tick_positions = np.linspace(0, n_improvements-1, 10, dtype=int)
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels([str(i) for i in tick_positions])
    
    # Plot 2: Regression cases
    if results['regressions']:
        n_regressions = len(results['regressions'])
        x_positions = np.arange(n_regressions)
        
        base_ppls = [r['base_avg_ppl'] for r in results['regressions']]
        enh_ppls = [r['enh_avg_ppl'] for r in results['regressions']]
        
        # Create bar plot
        width = 0.4
        bars1 = ax2.bar(x_positions - width/2, base_ppls, width, label='Base Model', 
                        color='lightblue', alpha=0.7)
        bars2 = ax2.bar(x_positions + width/2, enh_ppls, width, label='Enhanced Model', 
                        color='lightcoral', alpha=0.7)
        
        ax2.set_xlabel('Sample Index (sorted by Base PPL)')
        ax2.set_ylabel('Average Perplexity (log scale)')
        ax2.set_yscale('log')
        ax2.set_title(f'Regression Cases: PPL Distribution (n={n_regressions})', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y', which='both')
        
        # Add a horizontal line at PPL=100 for reference
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='PPL=100')
        
        # Limit x-axis ticks for readability
        if n_regressions > 20:
            tick_positions = np.linspace(0, n_regressions-1, 10, dtype=int)
            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels([str(i) for i in tick_positions])
    
    plt.suptitle('PPL Distribution Analysis: Base vs Enhanced Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'ppl_distribution_sorted.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"PPL distribution plot saved to {output_path}")
    
    # Create additional line plot for better trend visualization
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Line plot for improvements
    if results['improvements']:
        n_improvements = len(results['improvements'])
        x_positions = np.arange(n_improvements)
        
        base_ppls = [r['base_avg_ppl'] for r in results['improvements']]
        enh_ppls = [r['enh_avg_ppl'] for r in results['improvements']]
        
        ax3.plot(x_positions, base_ppls, label='Base Model', color='blue', 
                marker='o', markersize=4, alpha=0.7, linewidth=2)
        ax3.plot(x_positions, enh_ppls, label='Enhanced Model', color='green', 
                marker='s', markersize=4, alpha=0.7, linewidth=2)
        
        ax3.set_xlabel('Sample Index (sorted by Base PPL)')
        ax3.set_ylabel('Average Perplexity (log scale)')
        ax3.set_yscale('log')
        ax3.set_title(f'Improvement Cases: PPL Trend (n={n_improvements})', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, which='both')
        
        # Add shaded area between lines
        ax3.fill_between(x_positions, base_ppls, enh_ppls, 
                        where=(np.array(enh_ppls) < np.array(base_ppls)), 
                        color='green', alpha=0.2, label='PPL Reduction')
        ax3.fill_between(x_positions, base_ppls, enh_ppls, 
                        where=(np.array(enh_ppls) >= np.array(base_ppls)), 
                        color='red', alpha=0.2, label='PPL Increase')
    
    # Line plot for regressions
    if results['regressions']:
        n_regressions = len(results['regressions'])
        x_positions = np.arange(n_regressions)
        
        base_ppls = [r['base_avg_ppl'] for r in results['regressions']]
        enh_ppls = [r['enh_avg_ppl'] for r in results['regressions']]
        
        ax4.plot(x_positions, base_ppls, label='Base Model', color='blue', 
                marker='o', markersize=4, alpha=0.7, linewidth=2)
        ax4.plot(x_positions, enh_ppls, label='Enhanced Model', color='red', 
                marker='s', markersize=4, alpha=0.7, linewidth=2)
        
        ax4.set_xlabel('Sample Index (sorted by Base PPL)')
        ax4.set_ylabel('Average Perplexity (log scale)')
        ax4.set_yscale('log')
        ax4.set_title(f'Regression Cases: PPL Trend (n={n_regressions})', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, which='both')
        
        # Add shaded area between lines
        ax4.fill_between(x_positions, base_ppls, enh_ppls, 
                        where=(np.array(enh_ppls) < np.array(base_ppls)), 
                        color='green', alpha=0.2, label='PPL Reduction')
        ax4.fill_between(x_positions, base_ppls, enh_ppls, 
                        where=(np.array(enh_ppls) >= np.array(base_ppls)), 
                        color='red', alpha=0.2, label='PPL Increase')
    
    plt.suptitle('PPL Trend Analysis: Base vs Enhanced Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path2 = os.path.join(output_dir, 'ppl_distribution_trends.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    logger.info(f"PPL trend plot saved to {output_path2}")
    
    # Create summary statistics plot
    create_summary_statistics_plot(results, output_dir)


def create_summary_statistics_plot(results, output_dir):
    """Create a plot showing summary statistics of PPL changes."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate statistics
    stats = {}
    
    if results['improvements']:
        imp_base_ppls = [r['base_avg_ppl'] for r in results['improvements']]
        imp_enh_ppls = [r['enh_avg_ppl'] for r in results['improvements']]
        imp_changes = [e - b for b, e in zip(imp_base_ppls, imp_enh_ppls)]
        
        stats['improvements'] = {
            'count': len(results['improvements']),
            'base_mean': np.mean(imp_base_ppls),
            'base_std': np.std(imp_base_ppls),
            'enh_mean': np.mean(imp_enh_ppls),
            'enh_std': np.std(imp_enh_ppls),
            'mean_change': np.mean(imp_changes),
            'improved_count': sum(1 for c in imp_changes if c < 0),
            'worsened_count': sum(1 for c in imp_changes if c > 0)
        }
    
    if results['regressions']:
        reg_base_ppls = [r['base_avg_ppl'] for r in results['regressions']]
        reg_enh_ppls = [r['enh_avg_ppl'] for r in results['regressions']]
        reg_changes = [e - b for b, e in zip(reg_base_ppls, reg_enh_ppls)]
        
        stats['regressions'] = {
            'count': len(results['regressions']),
            'base_mean': np.mean(reg_base_ppls),
            'base_std': np.std(reg_base_ppls),
            'enh_mean': np.mean(reg_enh_ppls),
            'enh_std': np.std(reg_enh_ppls),
            'mean_change': np.mean(reg_changes),
            'improved_count': sum(1 for c in reg_changes if c < 0),
            'worsened_count': sum(1 for c in reg_changes if c > 0)
        }
    
    # Create grouped bar chart
    categories = ['Improvements', 'Regressions']
    base_means = []
    base_stds = []
    enh_means = []
    enh_stds = []
    
    for cat in ['improvements', 'regressions']:
        if cat in stats:
            base_means.append(stats[cat]['base_mean'])
            base_stds.append(stats[cat]['base_std'])
            enh_means.append(stats[cat]['enh_mean'])
            enh_stds.append(stats[cat]['enh_std'])
        else:
            base_means.append(0)
            base_stds.append(0)
            enh_means.append(0)
            enh_stds.append(0)
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_means, width, yerr=base_stds, 
                   label='Base Model', color='lightblue', alpha=0.7, capsize=5)
    bars2 = ax.bar(x + width/2, enh_means, width, yerr=enh_stds, 
                   label='Enhanced Model', color=['lightgreen', 'lightcoral'], alpha=0.7, capsize=5)
    
    ax.set_ylabel('Average Perplexity')
    ax.set_title('PPL Summary Statistics: Mean ± Std', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Add count information
    y_max = ax.get_ylim()[1]
    for i, cat in enumerate(['improvements', 'regressions']):
        if cat in stats:
            text = f"n={stats[cat]['count']}\n"
            text += f"PPL↓: {stats[cat]['improved_count']}\n"
            text += f"PPL↑: {stats[cat]['worsened_count']}"
            ax.text(i, y_max * 0.95, text, ha='center', va='top', 
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'ppl_summary_statistics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Summary statistics plot saved to {output_path}")
    
    # Print detailed statistics
    logger.info("\n=== Detailed PPL Statistics ===")
    if 'improvements' in stats:
        s = stats['improvements']
        logger.info(f"\nImprovement Cases (n={s['count']}):")
        logger.info(f"  Base PPL: {s['base_mean']:.2f} ± {s['base_std']:.2f}")
        logger.info(f"  Enhanced PPL: {s['enh_mean']:.2f} ± {s['enh_std']:.2f}")
        logger.info(f"  PPL Change: {s['mean_change']:.2f}")
        logger.info(f"  Cases with PPL reduction: {s['improved_count']} ({s['improved_count']/s['count']*100:.1f}%)")
        logger.info(f"  Cases with PPL increase: {s['worsened_count']} ({s['worsened_count']/s['count']*100:.1f}%)")
    
    if 'regressions' in stats:
        s = stats['regressions']
        logger.info(f"\nRegression Cases (n={s['count']}):")
        logger.info(f"  Base PPL: {s['base_mean']:.2f} ± {s['base_std']:.2f}")
        logger.info(f"  Enhanced PPL: {s['enh_mean']:.2f} ± {s['enh_std']:.2f}")
        logger.info(f"  PPL Change: {s['mean_change']:.2f}")
        logger.info(f"  Cases with PPL reduction: {s['improved_count']} ({s['improved_count']/s['count']*100:.1f}%)")
        logger.info(f"  Cases with PPL increase: {s['worsened_count']} ({s['worsened_count']/s['count']*100:.1f}%)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='PPL distribution analysis for improvement and regression cases')
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser.add_argument('--base', default=os.path.join(project_root, 'saves/base.jsonl'), 
                       help='Path to base.jsonl file')
    parser.add_argument('--enhanced', default=os.path.join(project_root, 'saves/enhanced.jsonl'), 
                       help='Path to enhanced.jsonl file')
    parser.add_argument('--output-dir', default=os.path.join(project_root, 'saves/ppl_distribution_output'), help='Output directory')
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                       help='Model to use for PPL analysis')
    parser.add_argument('--n-tokens', type=int, default=100, help='Number of tokens to analyze')
    parser.add_argument('--response-only', action='store_true', 
                       help='Analyze only response text, excluding problem')
    
    args = parser.parse_args()
    
    # Load and match data
    logger.info("Loading and matching data...")
    base_data, enhanced_data, improvements, regressions = load_and_match_data(
        args.base, args.enhanced
    )
    
    # Initialize analyzer
    logger.info("Initializing token analyzer...")
    analyzer = TokenAnalyzer(args.model)
    
    # Analyze PPL distribution
    include_problem = not args.response_only
    logger.info(f"Analysis mode: {'Problem + Response' if include_problem else 'Response only'}")
    
    results = analyze_ppl_distribution(
        analyzer, base_data, enhanced_data, 
        improvements, regressions, 
        args.n_tokens, include_problem
    )
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_ppl_distribution_plots(results, args.output_dir)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'ppl_distribution_results.json')
    
    # Convert results to JSON-serializable format
    json_results = {
        'improvements': [
            {
                'base_idx': r['base_idx'],
                'enh_idx': r['enh_idx'],
                'base_avg_ppl': float(r['base_avg_ppl']),
                'enh_avg_ppl': float(r['enh_avg_ppl'])
            }
            for r in results['improvements']
        ],
        'regressions': [
            {
                'base_idx': r['base_idx'],
                'enh_idx': r['enh_idx'],
                'base_avg_ppl': float(r['base_avg_ppl']),
                'enh_avg_ppl': float(r['enh_avg_ppl'])
            }
            for r in results['regressions']
        ]
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info("PPL distribution analysis completed!")


if __name__ == "__main__":
    main()