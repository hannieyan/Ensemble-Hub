#!/usr/bin/env python3
"""
Single Sample Token Analysis Script
Based on complete_token_analysis.py, but allows selecting specific samples
for detailed token-level analysis visualization.
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
from scipy.stats import entropy
from tqdm import tqdm
from collections import defaultdict
import logging
import subprocess
import tempfile
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
from complete_token_analysis import TokenAnalyzer, load_and_match_data, analyze_samples


def create_single_sample_visualizations(results, sample_count, start_index, output_dir, n_tokens=100):
    """Create visualizations for specific samples."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which samples to show - BOTH types if they have enough samples
    show_improvements = False
    show_regressions = False
    
    improvement_start_idx = None
    improvement_end_idx = None
    regression_start_idx = None
    regression_end_idx = None
    
    # Calculate sample ranges
    total_improvements = len(results['improvements']['base']) if results['improvements']['base'] else 0
    total_regressions = len(results['regressions']['base']) if results['regressions']['base'] else 0
    
    logger.info(f"Total improvements: {total_improvements}, Total regressions: {total_regressions}")
    logger.info(f"Requested: {sample_count} samples starting from index {start_index}")
    
    # Check if improvements have enough samples
    if start_index < total_improvements:
        show_improvements = True
        improvement_start_idx = start_index
        improvement_end_idx = min(start_index + sample_count, total_improvements)
        logger.info(f"Will show improvements from index {improvement_start_idx} to {improvement_end_idx-1}")
    
    # Check if regressions have enough samples (independent of improvements)
    if start_index < total_regressions:
        show_regressions = True
        regression_start_idx = start_index
        regression_end_idx = min(start_index + sample_count, total_regressions)
        logger.info(f"Will show regressions from index {regression_start_idx} to {regression_end_idx-1}")
    
    if not show_improvements and not show_regressions:
        logger.warning(f"Start index {start_index} is beyond all available samples!")
        return
    
    # Create figure layout based on what we're showing
    fig_height = 10
    if show_improvements and show_regressions:
        fig, axes = plt.subplots(2, 2, figsize=(16, fig_height))
        imp_ppl_ax = axes[0, 0]
        imp_ent_ax = axes[0, 1]
        reg_ppl_ax = axes[1, 0]
        reg_ent_ax = axes[1, 1]
    elif show_improvements:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        imp_ppl_ax = axes[0]
        imp_ent_ax = axes[1]
        reg_ppl_ax = None
        reg_ent_ax = None
    elif show_regressions:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        reg_ppl_ax = axes[0]
        reg_ent_ax = axes[1]
        imp_ppl_ax = None
        imp_ent_ax = None
    
    # Plot improvements
    if show_improvements and imp_ppl_ax is not None:
        # PPL plot for improvements
        for idx in range(improvement_start_idx, improvement_end_idx):
            base_result = results['improvements']['base'][idx]
            enh_result = results['improvements']['enhanced'][idx]
            
            sample_label = f"Improvement {idx}"
            
            # Plot PPL
            x_positions = range(len(base_result['ppls']))
            imp_ppl_ax.plot(x_positions, base_result['ppls'], 
                           label=f'{sample_label} - Base', 
                           color='blue', alpha=0.7, linewidth=2, marker='o', markersize=3)
            
            x_positions_enh = range(len(enh_result['ppls'][:len(base_result['ppls'])]))
            imp_ppl_ax.plot(x_positions_enh, enh_result['ppls'][:len(base_result['ppls'])], 
                           label=f'{sample_label} - Enhanced', 
                           color='green', alpha=0.7, linewidth=2, marker='s', markersize=3)
        
        imp_ppl_ax.set_xlabel('Token Position')
        imp_ppl_ax.set_ylabel('Perplexity (log scale)')
        imp_ppl_ax.set_yscale('log')
        imp_ppl_ax.set_title(f'Improvement Cases: PPL by Token Position\n(Samples {improvement_start_idx}-{improvement_end_idx-1})')
        imp_ppl_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        imp_ppl_ax.grid(True, alpha=0.3, which='both')
        
        # Entropy plot for improvements
        for idx in range(improvement_start_idx, improvement_end_idx):
            base_result = results['improvements']['base'][idx]
            enh_result = results['improvements']['enhanced'][idx]
            
            sample_label = f"Improvement {idx}"
            
            # Plot Entropy
            x_positions = range(len(base_result['entropies']))
            imp_ent_ax.plot(x_positions, base_result['entropies'], 
                           label=f'{sample_label} - Base', 
                           color='blue', alpha=0.7, linewidth=2, marker='o', markersize=3)
            
            x_positions_enh = range(len(enh_result['entropies'][:len(base_result['entropies'])]))
            imp_ent_ax.plot(x_positions_enh, enh_result['entropies'][:len(base_result['entropies'])], 
                           label=f'{sample_label} - Enhanced', 
                           color='green', alpha=0.7, linewidth=2, marker='s', markersize=3)
        
        imp_ent_ax.set_xlabel('Token Position')
        imp_ent_ax.set_ylabel('Entropy (bits)')
        imp_ent_ax.set_title(f'Improvement Cases: Entropy by Token Position\n(Samples {improvement_start_idx}-{improvement_end_idx-1})')
        imp_ent_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        imp_ent_ax.grid(True, alpha=0.3)
    
    # Plot regressions
    if show_regressions and reg_ppl_ax is not None:
        # PPL plot for regressions
        for idx in range(regression_start_idx, regression_end_idx):
            base_result = results['regressions']['base'][idx]
            enh_result = results['regressions']['enhanced'][idx]
            
            sample_label = f"Regression {idx}"
            
            # Plot PPL
            x_positions = range(len(base_result['ppls']))
            reg_ppl_ax.plot(x_positions, base_result['ppls'], 
                           label=f'{sample_label} - Base', 
                           color='blue', alpha=0.7, linewidth=2, marker='o', markersize=3)
            
            x_positions_enh = range(len(enh_result['ppls'][:len(base_result['ppls'])]))
            reg_ppl_ax.plot(x_positions_enh, enh_result['ppls'][:len(base_result['ppls'])], 
                           label=f'{sample_label} - Enhanced', 
                           color='red', alpha=0.7, linewidth=2, marker='s', markersize=3)
        
        reg_ppl_ax.set_xlabel('Token Position')
        reg_ppl_ax.set_ylabel('Perplexity (log scale)')
        reg_ppl_ax.set_yscale('log')
        reg_ppl_ax.set_title(f'Regression Cases: PPL by Token Position\n(Samples {regression_start_idx}-{regression_end_idx-1})')
        reg_ppl_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        reg_ppl_ax.grid(True, alpha=0.3, which='both')
        
        # Entropy plot for regressions
        for idx in range(regression_start_idx, regression_end_idx):
            base_result = results['regressions']['base'][idx]
            enh_result = results['regressions']['enhanced'][idx]
            
            sample_label = f"Regression {idx}"
            
            # Plot Entropy
            x_positions = range(len(base_result['entropies']))
            reg_ent_ax.plot(x_positions, base_result['entropies'], 
                           label=f'{sample_label} - Base', 
                           color='blue', alpha=0.7, linewidth=2, marker='o', markersize=3)
            
            x_positions_enh = range(len(enh_result['entropies'][:len(base_result['entropies'])]))
            reg_ent_ax.plot(x_positions_enh, enh_result['entropies'][:len(base_result['entropies'])], 
                           label=f'{sample_label} - Enhanced', 
                           color='red', alpha=0.7, linewidth=2, marker='s', markersize=3)
        
        reg_ent_ax.set_xlabel('Token Position')
        reg_ent_ax.set_ylabel('Entropy (bits)')
        reg_ent_ax.set_title(f'Regression Cases: Entropy by Token Position\n(Samples {regression_start_idx}-{regression_end_idx-1})')
        reg_ent_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        reg_ent_ax.grid(True, alpha=0.3)
    
    # Add empty plots for missing sections if needed
    if show_improvements and not show_regressions and len(axes.shape) == 2:
        # Hide regression plots if they exist but we don't need them
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)
    
    # Overall title
    title_parts = []
    if show_improvements:
        title_parts.append(f"Improvements: {improvement_start_idx}-{improvement_end_idx-1}")
    if show_regressions:
        title_parts.append(f"Regressions: {regression_start_idx}-{regression_end_idx-1}")
    
    plt.suptitle(f'Single Sample Token Analysis ({", ".join(title_parts)})', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    
    output_path = os.path.join(output_dir, f'single_sample_analysis_start{start_index}_count{sample_count}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Single sample analysis plot saved to {output_path}")
    
    # Save sample details
    save_sample_details(results, sample_count, start_index, output_dir, 
                       show_improvements, show_regressions,
                       improvement_start_idx, improvement_end_idx,
                       regression_start_idx, regression_end_idx)


def save_sample_details(results, sample_count, start_index, output_dir,
                       show_improvements, show_regressions,
                       improvement_start_idx, improvement_end_idx,
                       regression_start_idx, regression_end_idx):
    """Save detailed information about the analyzed samples."""
    
    sample_details = {
        'parameters': {
            'sample_count': sample_count,
            'start_index': start_index,
            'total_improvements': len(results['improvements']['base']) if results['improvements']['base'] else 0,
            'total_regressions': len(results['regressions']['base']) if results['regressions']['base'] else 0
        },
        'analyzed_samples': {
            'improvements': [],
            'regressions': []
        }
    }
    
    # Save improvement sample details
    if show_improvements:
        for idx in range(improvement_start_idx, improvement_end_idx):
            base_result = results['improvements']['base'][idx]
            enh_result = results['improvements']['enhanced'][idx]
            
            sample_info = {
                'sample_index': idx,
                'improvement_index': idx,
                'base_stats': {
                    'avg_ppl': float(base_result['avg_ppl']),
                    'avg_entropy': float(base_result['avg_entropy']),
                    'token_count': len(base_result['ppls'])
                },
                'enhanced_stats': {
                    'avg_ppl': float(enh_result['avg_ppl']),
                    'avg_entropy': float(enh_result['avg_entropy']),
                    'token_count': len(enh_result['ppls'])
                },
                'ppl_change': float(enh_result['avg_ppl'] - base_result['avg_ppl']),
                'entropy_change': float(enh_result['avg_entropy'] - base_result['avg_entropy'])
            }
            sample_details['analyzed_samples']['improvements'].append(sample_info)
    
    # Save regression sample details
    if show_regressions:
        for idx in range(regression_start_idx, regression_end_idx):
            base_result = results['regressions']['base'][idx]
            enh_result = results['regressions']['enhanced'][idx]
            
            sample_info = {
                'sample_index': idx,
                'regression_index': idx,
                'base_stats': {
                    'avg_ppl': float(base_result['avg_ppl']),
                    'avg_entropy': float(base_result['avg_entropy']),
                    'token_count': len(base_result['ppls'])
                },
                'enhanced_stats': {
                    'avg_ppl': float(enh_result['avg_ppl']),
                    'avg_entropy': float(enh_result['avg_entropy']),
                    'token_count': len(enh_result['ppls'])
                },
                'ppl_change': float(enh_result['avg_ppl'] - base_result['avg_ppl']),
                'entropy_change': float(enh_result['avg_entropy'] - base_result['avg_entropy'])
            }
            sample_details['analyzed_samples']['regressions'].append(sample_info)
    
    # Save to file
    details_path = os.path.join(output_dir, f'sample_details_start{start_index}_count{sample_count}.json')
    with open(details_path, 'w') as f:
        json.dump(sample_details, f, indent=2)
    
    logger.info(f"Sample details saved to {details_path}")
    
    # Print summary
    logger.info(f"\n=== Sample Analysis Summary ===")
    logger.info(f"Requested: {sample_count} samples starting from index {start_index}")
    
    if show_improvements:
        logger.info(f"Analyzed improvements: indices {improvement_start_idx}-{improvement_end_idx-1}")
        for sample in sample_details['analyzed_samples']['improvements']:
            logger.info(f"  Improvement {sample['improvement_index']}: PPL change = {sample['ppl_change']:.2f}, "
                       f"Entropy change = {sample['entropy_change']:.3f}")
    
    if show_regressions:
        logger.info(f"Analyzed regressions: indices {regression_start_idx}-{regression_end_idx-1}")
        for sample in sample_details['analyzed_samples']['regressions']:
            logger.info(f"  Regression {sample['regression_index']}: PPL change = {sample['ppl_change']:.2f}, "
                       f"Entropy change = {sample['entropy_change']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Single sample token analysis with controllable sample selection')
    
    # Get project root directory (parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser.add_argument('--base', default=os.path.join(project_root, 'saves/base.jsonl'), 
                       help='Path to base.jsonl file')
    parser.add_argument('--enhanced', default=os.path.join(project_root, 'saves/enhanced.jsonl'), 
                       help='Path to enhanced.jsonl file')
    parser.add_argument('--output-dir', default=os.path.join(project_root, 'saves/single_sample_analysis_output'), help='Output directory')
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                       help='Model to use for token analysis')
    parser.add_argument('--n-tokens', type=int, default=100, help='Number of tokens to analyze')
    parser.add_argument('--sample-count', type=int, default=1, 
                       help='Number of samples to analyze (default: 1)')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Starting sample index (0-based, across all samples)')
    parser.add_argument('--include-problem', action='store_true', default=True, 
                       help='Include problem text in token analysis (default: True)')
    parser.add_argument('--response-only', action='store_true', 
                       help='Analyze only response text, excluding problem')
    parser.add_argument('--entropy-method', choices=['full', 'top-k'], default='full',
                       help='Entropy calculation method: full (all tokens) or top-k (top-k tokens only)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.sample_count < 1:
        logger.error("Sample count must be at least 1")
        return
    
    if args.start_index < 0:
        logger.error("Start index must be non-negative")
        return
    
    # Load and match data
    base_data, enhanced_data, improvements, regressions = load_and_match_data(
        args.base, args.enhanced
    )
    
    logger.info(f"Found {len(improvements)} improvements and {len(regressions)} regressions")
    
    total_samples = len(improvements) + len(regressions)
    if args.start_index >= total_samples:
        logger.error(f"Start index {args.start_index} is beyond available samples (0-{total_samples-1})")
        return
    
    # Initialize analyzer
    analyzer = TokenAnalyzer(args.model)
    
    # Determine if we should include problem text
    include_problem = not args.response_only if args.response_only else args.include_problem
    
    logger.info(f"Token analysis mode: {'Problem + Response' if include_problem else 'Response only'}")
    
    # Analyze samples (get all samples first)
    results = analyze_samples(
        analyzer, base_data, enhanced_data, 
        improvements, regressions, 
        None, args.n_tokens, include_problem, args.entropy_method
    )
    
    # Create single sample visualizations
    create_single_sample_visualizations(results, args.sample_count, args.start_index, 
                                      args.output_dir, args.n_tokens)
    
    logger.info("Single sample token analysis completed!")


if __name__ == "__main__":
    main()