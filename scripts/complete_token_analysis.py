#!/usr/bin/env python3
"""
Complete Token Analysis Script
Analyzes base and enhanced model responses, identifies improvements and regressions,
and performs token-level analysis using DeepSeek model.
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TokenAnalyzer:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", device=None):
        """Initialize the token analyzer with the specified model."""
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Error loading model with default settings: {e}")
            logger.info("Trying with 8-bit quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto"
            )
        
        self.model.eval()
    
    def compute_token_metrics(self, text, n_tokens=100, entropy_method='full'):
        """Compute perplexity and entropy for the first n tokens."""
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = inputs["input_ids"].to(self.device)
            
            # Limit to n_tokens + 1 for prediction
            if input_ids.shape[1] > n_tokens + 1:
                input_ids = input_ids[:, :n_tokens + 1]
            
            if input_ids.shape[1] <= 1:
                return [], [], []
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
            
            # Calculate metrics
            tokens = []
            ppls = []
            entropies = []
            
            for i in range(min(n_tokens, logits.shape[1] - 1)):
                current_logits = logits[0, i, :]
                probs = torch.softmax(current_logits, dim=-1)
                actual_token = input_ids[0, i + 1].item()
                
                # Perplexity
                token_prob = probs[actual_token].item()
                ppl = 1.0 / token_prob if token_prob > 0 else 1000.0  # Cap at 1000
                
                # Entropy calculation with different methods
                probs_np = probs.cpu().numpy().astype(np.float64)  # Use higher precision
                
                if entropy_method == 'top-k':
                    # Use only top-1000 probabilities to avoid numerical issues
                    top_k = 1000
                    top_indices = np.argpartition(probs_np, -top_k)[-top_k:]
                    probs_np = probs_np[top_indices]
                    # Normalize the top-k probabilities
                    probs_np = probs_np / probs_np.sum()
                    # Remove very small probabilities
                    probs_np = probs_np[probs_np > 1e-8]
                else:
                    # Full method - use all probabilities but filter small ones
                    probs_np = probs_np[probs_np > 1e-10]
                
                if len(probs_np) == 0 or np.sum(probs_np) == 0:
                    token_entropy = 0.0
                else:
                    # Normalize to ensure sum = 1
                    probs_np = probs_np / probs_np.sum()
                    
                    # Calculate entropy with natural log first, then convert
                    log_probs = np.log(probs_np)
                    token_entropy = -np.sum(probs_np * log_probs)
                    
                    # Convert to bits (base 2) for interpretability
                    token_entropy = token_entropy / np.log(2)
                    
                    # Check for invalid values
                    if np.isnan(token_entropy) or np.isinf(token_entropy) or token_entropy < 0:
                        token_entropy = 0.0
                
                
                # Token text
                token_text = self.tokenizer.decode([actual_token])
                
                ppls.append(min(ppl, 1000.0))
                entropies.append(token_entropy)
                tokens.append(token_text)
            
            return tokens, ppls, entropies
            
        except Exception as e:
            logger.error(f"Error computing token metrics: {e}")
            return [], [], []
    
    def compute_answer_token_metrics(self, problem_text, answer_text, n_tokens=100, entropy_method='full'):
        """Compute perplexity and entropy for the first n tokens of the answer only, 
        but with problem context for better prediction."""
        try:
            # Create full text for context
            full_text = f"Problem: {problem_text}\n\nSolution: {answer_text}"
            
            # Tokenize both parts separately to find answer start position
            problem_part = f"Problem: {problem_text}\n\nSolution: "
            problem_inputs = self.tokenizer(problem_part, return_tensors="pt", truncation=False)
            problem_length = problem_inputs["input_ids"].shape[1]
            
            # Tokenize full text
            full_inputs = self.tokenizer(full_text, return_tensors="pt", truncation=False)
            full_input_ids = full_inputs["input_ids"].to(self.device)
            
            # Calculate how many answer tokens we need (limited by n_tokens)
            total_length = full_input_ids.shape[1]
            answer_length = total_length - problem_length
            actual_answer_tokens = min(n_tokens, answer_length)
            
            # We need problem context + answer tokens for prediction
            # But we only analyze the answer part
            needed_length = problem_length + actual_answer_tokens
            if total_length > needed_length:
                input_ids = full_input_ids[:, :needed_length]
            else:
                input_ids = full_input_ids
            
            if input_ids.shape[1] <= problem_length:
                # Not enough tokens for analysis
                return [], [], []
            
            # Get model outputs for the full context
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
            
            # Calculate metrics only for answer tokens
            tokens = []
            ppls = []
            entropies = []
            
            # Start analysis from the first answer token
            start_idx = problem_length - 1  # -1 because we predict the next token
            end_idx = min(problem_length - 1 + actual_answer_tokens, logits.shape[1] - 1)
            
            for i in range(start_idx, end_idx):
                current_logits = logits[0, i, :]
                probs = torch.softmax(current_logits, dim=-1)
                actual_token = input_ids[0, i + 1].item()
                
                # Perplexity
                token_prob = probs[actual_token].item()
                ppl = 1.0 / token_prob if token_prob > 0 else 1000.0  # Cap at 1000
                
                # Entropy calculation with different methods
                probs_np = probs.cpu().numpy().astype(np.float64)  # Use higher precision
                
                if entropy_method == 'top-k':
                    # Use only top-1000 probabilities to avoid numerical issues
                    top_k = 1000
                    top_indices = np.argpartition(probs_np, -top_k)[-top_k:]
                    probs_np = probs_np[top_indices]
                    # Normalize the top-k probabilities
                    probs_np = probs_np / probs_np.sum()
                    # Remove very small probabilities
                    probs_np = probs_np[probs_np > 1e-8]
                else:
                    # Full method - use all probabilities but filter small ones
                    probs_np = probs_np[probs_np > 1e-10]
                
                if len(probs_np) == 0 or np.sum(probs_np) == 0:
                    token_entropy = 0.0
                else:
                    # Normalize to ensure sum = 1
                    probs_np = probs_np / probs_np.sum()
                    
                    # Calculate entropy with natural log first, then convert
                    log_probs = np.log(probs_np)
                    token_entropy = -np.sum(probs_np * log_probs)
                    
                    # Convert to bits (base 2) for interpretability
                    token_entropy = token_entropy / np.log(2)
                    
                    # Check for invalid values
                    if np.isnan(token_entropy) or np.isinf(token_entropy) or token_entropy < 0:
                        token_entropy = 0.0
                
                # Token text
                token_text = self.tokenizer.decode([actual_token])
                
                ppls.append(min(ppl, 1000.0))
                entropies.append(token_entropy)
                tokens.append(token_text)
            
            return tokens, ppls, entropies
            
        except Exception as e:
            logger.error(f"Error computing answer token metrics: {e}")
            return [], [], []


def run_evaluation(jsonl_path, output_dir=None):
    """Run eval_acc_lm_eval.py to generate evaluation results for a jsonl file."""
    if output_dir is None:
        output_dir = os.path.dirname(jsonl_path) or '.'
    
    # Create a temporary output file for the evaluation results
    base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    eval_results_path = os.path.join(output_dir, f"{base_name}_temp_eval_results.jsonl")
    
    # Run the evaluation script - use absolute path to avoid path issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script_path = os.path.join(script_dir, "eval_acc_lm_eval.py")
    cmd = ["python", eval_script_path, jsonl_path]
    
    logger.info(f"Running evaluation: {' '.join(cmd)}")
    
    try:
        # Set working directory to the project root (parent of scripts directory)
        project_root = os.path.dirname(script_dir)
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=project_root)
        logger.info("Evaluation completed successfully")
        
        # Parse the output to extract detailed results
        
        # Create evaluation results by parsing the original jsonl file and adding correctness info
        detailed_results = []
        
        with jsonlines.open(jsonl_path) as reader:
            data_items = list(reader)
        
        # For now, we'll run a simplified evaluation to get correctness info
        # This is a basic implementation - you might need to adjust based on actual output format
        for idx, item in enumerate(data_items):
            # Create a basic evaluation entry
            eval_entry = {
                "index": idx,
                "doc_id": idx,  # Use index as doc_id if not available
                "correct": False  # This will be updated based on actual evaluation
            }
            
            # Try to determine correctness from the item
            if 'doc' in item and 'solution' in item['doc'] and 'resps' in item and item['resps']:
                # Import grader from the scripts directory - use absolute path
                import sys
                script_dir = os.path.dirname(os.path.abspath(__file__))
                if script_dir not in sys.path:
                    sys.path.append(script_dir)
                
                try:
                    from grader import grade_answer
                    solution_text = item['doc']['solution']
                    response_text = item['resps'][0][0] if item['resps'][0] else ""
                    
                    # Extract boxed content from both
                    def extract_boxed_content(text):
                        start = text.find(r'\boxed{')
                        if start == -1:
                            return ""
                        i = start + len(r'\boxed{')
                        depth = 1
                        content = ""
                        while i < len(text) and depth > 0:
                            if text[i] == '{':
                                depth += 1
                            elif text[i] == '}':
                                depth -= 1
                            if depth > 0:
                                content += text[i]
                            i += 1
                        return content.strip()
                    
                    solution_boxed = extract_boxed_content(solution_text)
                    response_boxed = extract_boxed_content(response_text)
                    
                    if solution_boxed and response_boxed:
                        eval_entry["correct"] = grade_answer(solution_boxed, response_boxed)
                    
                except Exception:
                    eval_entry["correct"] = False
            
            detailed_results.append(eval_entry)
        
        # Save detailed results
        with jsonlines.open(eval_results_path, mode='w') as writer:
            for result in detailed_results:
                writer.write(result)
        
        logger.info(f"Saved evaluation results to {eval_results_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    
    if not os.path.exists(eval_results_path):
        raise FileNotFoundError(f"Expected evaluation results file not found: {eval_results_path}")
    
    return eval_results_path


def load_and_match_data(base_path, enhanced_path):
    """Load base and enhanced data and match them by problem."""
    # Load response data
    logger.info("Loading response data...")
    with jsonlines.open(base_path) as reader:
        base_data = list(reader)
    
    with jsonlines.open(enhanced_path) as reader:
        enhanced_data = list(reader)
    
    logger.info(f"Loaded {len(base_data)} base and {len(enhanced_data)} enhanced responses")
    
    # Auto-generate evaluation results
    logger.info("Generating evaluation results...")
    temp_dir = tempfile.mkdtemp()
    
    try:
        base_eval_path = run_evaluation(base_path, temp_dir)
        enhanced_eval_path = run_evaluation(enhanced_path, temp_dir)
        
        # Load evaluation results
        base_detailed = {}
        enhanced_detailed = {}
        
        with jsonlines.open(base_eval_path) as reader:
            for obj in reader:
                base_detailed[obj["index"]] = obj
        
        with jsonlines.open(enhanced_eval_path) as reader:
            for obj in reader:
                enhanced_detailed[obj["index"]] = obj
        
        # Create mapping
        base_problems = {base_data[i]['doc']['problem']: i for i in range(len(base_data))}
        improvements = []
        regressions = []
        
        for enh_idx, enh_item in enumerate(enhanced_data):
            enh_problem = enh_item['doc']['problem']
            if enh_problem in base_problems:
                base_idx = base_problems[enh_problem]
                
                # Use evaluation results
                if base_idx in base_detailed and enh_idx in enhanced_detailed:
                    base_correct = base_detailed[base_idx]["correct"]
                    enhanced_correct = enhanced_detailed[enh_idx]["correct"]
                    
                    if enhanced_correct and not base_correct:
                        improvements.append((base_idx, enh_idx))
                    elif base_correct and not enhanced_correct:
                        regressions.append((base_idx, enh_idx))
        
        logger.info(f"Found {len(improvements)} improvements and {len(regressions)} regressions")
        
    finally:
        # Clean up temporary evaluation files
        import shutil
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary evaluation files")
    
    return base_data, enhanced_data, improvements, regressions


def save_samples(base_data, enhanced_data, improvements, regressions, output_dir):
    """Save improvement and regression samples to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save improvements
    improvement_samples = []
    for base_idx, enh_idx in improvements:
        sample = {
            'doc_id': base_idx,  # or use from evaluation if available
            'problem': base_data[base_idx]['doc']['problem'],
            'base_response': base_data[base_idx]['resps'][0][0],
            'enhanced_response': enhanced_data[enh_idx]['resps'][0][0],
            'base_correct': False,
            'enhanced_correct': True
        }
        improvement_samples.append(sample)
    
    with jsonlines.open(os.path.join(output_dir, "improvements_simple.jsonl"), mode='w') as writer:
        for sample in improvement_samples:
            writer.write(sample)
    
    # Save regressions
    regression_samples = []
    for base_idx, enh_idx in regressions:
        sample = {
            'doc_id': base_idx,
            'problem': base_data[base_idx]['doc']['problem'],
            'base_response': base_data[base_idx]['resps'][0][0],
            'enhanced_response': enhanced_data[enh_idx]['resps'][0][0],
            'base_correct': True,
            'enhanced_correct': False
        }
        regression_samples.append(sample)
    
    with jsonlines.open(os.path.join(output_dir, "regressions_simple.jsonl"), mode='w') as writer:
        for sample in regression_samples:
            writer.write(sample)
    
    logger.info(f"Saved {len(improvement_samples)} improvements and {len(regression_samples)} regressions")
    
    return improvement_samples, regression_samples


def analyze_samples(analyzer, base_data, enhanced_data, improvements, regressions, 
                   sample_size=None, n_tokens=100, include_problem=True, entropy_method='full'):
    """Analyze token metrics for improvement and regression samples."""
    results = {
        'improvements': {'base': [], 'enhanced': []},
        'regressions': {'base': [], 'enhanced': []}
    }
    
    # Sample if needed
    if sample_size:
        improvements = improvements[:sample_size] if len(improvements) > sample_size else improvements
        regressions = regressions[:sample_size] if len(regressions) > sample_size else regressions
    
    logger.info(f"Analyzing {len(improvements)} improvements and {len(regressions)} regressions")
    
    # Analyze improvements
    logger.info("Analyzing improvement cases...")
    for base_idx, enh_idx in tqdm(improvements, desc="Improvements"):
        # Get the problem text
        problem_text = base_data[base_idx]['doc']['problem']
        
        # Create input text based on include_problem parameter
        base_resp = base_data[base_idx]["resps"][0][0]
        enh_resp = enhanced_data[enh_idx]["resps"][0][0]
        
        if include_problem:
            # Use the new method that analyzes only answer tokens but with problem context
            base_tokens, base_ppls, base_entropies = analyzer.compute_answer_token_metrics(problem_text, base_resp, n_tokens, entropy_method)
            enh_tokens, enh_ppls, enh_entropies = analyzer.compute_answer_token_metrics(problem_text, enh_resp, n_tokens, entropy_method)
        else:
            # Analyze only the response text
            base_tokens, base_ppls, base_entropies = analyzer.compute_token_metrics(base_resp, n_tokens, entropy_method)
            enh_tokens, enh_ppls, enh_entropies = analyzer.compute_token_metrics(enh_resp, n_tokens, entropy_method)
        
        if base_tokens and enh_tokens:
            results['improvements']['base'].append({
                'idx': base_idx,
                'tokens': base_tokens,
                'ppls': base_ppls,
                'entropies': base_entropies,
                'avg_ppl': np.mean(base_ppls),
                'avg_entropy': np.mean(base_entropies)
            })
            
            results['improvements']['enhanced'].append({
                'idx': enh_idx,
                'tokens': enh_tokens,
                'ppls': enh_ppls,
                'entropies': enh_entropies,
                'avg_ppl': np.mean(enh_ppls),
                'avg_entropy': np.mean(enh_entropies)
            })
    
    # Analyze regressions
    logger.info("Analyzing regression cases...")
    for base_idx, enh_idx in tqdm(regressions, desc="Regressions"):
        # Get the problem text
        problem_text = base_data[base_idx]['doc']['problem']
        
        # Create input text based on include_problem parameter
        base_resp = base_data[base_idx]["resps"][0][0]
        enh_resp = enhanced_data[enh_idx]["resps"][0][0]
        
        if include_problem:
            # Use the new method that analyzes only answer tokens but with problem context
            base_tokens, base_ppls, base_entropies = analyzer.compute_answer_token_metrics(problem_text, base_resp, n_tokens, entropy_method)
            enh_tokens, enh_ppls, enh_entropies = analyzer.compute_answer_token_metrics(problem_text, enh_resp, n_tokens, entropy_method)
        else:
            # Analyze only the response text
            base_tokens, base_ppls, base_entropies = analyzer.compute_token_metrics(base_resp, n_tokens, entropy_method)
            enh_tokens, enh_ppls, enh_entropies = analyzer.compute_token_metrics(enh_resp, n_tokens, entropy_method)
        
        if base_tokens and enh_tokens:
            results['regressions']['base'].append({
                'idx': base_idx,
                'tokens': base_tokens,
                'ppls': base_ppls,
                'entropies': base_entropies,
                'avg_ppl': np.mean(base_ppls),
                'avg_entropy': np.mean(base_entropies)
            })
            
            results['regressions']['enhanced'].append({
                'idx': enh_idx,
                'tokens': enh_tokens,
                'ppls': enh_ppls,
                'entropies': enh_entropies,
                'avg_ppl': np.mean(enh_ppls),
                'avg_entropy': np.mean(enh_entropies)
            })
    
    return results


def calculate_statistics(results):
    """Calculate summary statistics from the analysis results."""
    stats = {}
    
    # Improvement statistics
    if results['improvements']['base']:
        imp_base_ppls = [r['avg_ppl'] for r in results['improvements']['base']]
        imp_enh_ppls = [r['avg_ppl'] for r in results['improvements']['enhanced']]
        imp_base_ents = [r['avg_entropy'] for r in results['improvements']['base']]
        imp_enh_ents = [r['avg_entropy'] for r in results['improvements']['enhanced']]
        
        stats['improvements'] = {
            'count': len(results['improvements']['base']),
            'base_avg_ppl': np.mean(imp_base_ppls),
            'enhanced_avg_ppl': np.mean(imp_enh_ppls),
            'base_avg_entropy': np.mean(imp_base_ents),
            'enhanced_avg_entropy': np.mean(imp_enh_ents),
            'ppl_change': np.mean(imp_enh_ppls) - np.mean(imp_base_ppls),
            'entropy_change': np.mean(imp_enh_ents) - np.mean(imp_base_ents),
            'base_ppls': imp_base_ppls,
            'enhanced_ppls': imp_enh_ppls,
            'base_entropies': imp_base_ents,
            'enhanced_entropies': imp_enh_ents
        }
    
    # Regression statistics
    if results['regressions']['base']:
        reg_base_ppls = [r['avg_ppl'] for r in results['regressions']['base']]
        reg_enh_ppls = [r['avg_ppl'] for r in results['regressions']['enhanced']]
        reg_base_ents = [r['avg_entropy'] for r in results['regressions']['base']]
        reg_enh_ents = [r['avg_entropy'] for r in results['regressions']['enhanced']]
        
        stats['regressions'] = {
            'count': len(results['regressions']['base']),
            'base_avg_ppl': np.mean(reg_base_ppls),
            'enhanced_avg_ppl': np.mean(reg_enh_ppls),
            'base_avg_entropy': np.mean(reg_base_ents),
            'enhanced_avg_entropy': np.mean(reg_enh_ents),
            'ppl_change': np.mean(reg_enh_ppls) - np.mean(reg_base_ppls),
            'entropy_change': np.mean(reg_enh_ents) - np.mean(reg_base_ents),
            'base_ppls': reg_base_ppls,
            'enhanced_ppls': reg_enh_ppls,
            'base_entropies': reg_base_ents,
            'enhanced_entropies': reg_enh_ents
        }
    
    return stats


def create_visualizations(stats, results, output_dir, n_tokens=100):
    """Create comprehensive visualizations of the analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Improvement Cases - Perplexity Scatter
    ax1 = axes[0, 0]
    if 'improvements' in stats:
        ax1.scatter(stats['improvements']['base_ppls'], 
                   stats['improvements']['enhanced_ppls'], 
                   alpha=0.6, color='green', s=50)
        max_val = max(max(stats['improvements']['base_ppls']), 
                     max(stats['improvements']['enhanced_ppls']))
        ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        ax1.set_xlabel('Base Average Perplexity')
        ax1.set_ylabel('Enhanced Average Perplexity')
        ax1.set_title('Improvement Cases: Perplexity Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 2. Improvement Cases - Entropy Scatter
    ax2 = axes[0, 1]
    if 'improvements' in stats:
        ax2.scatter(stats['improvements']['base_entropies'], 
                   stats['improvements']['enhanced_entropies'], 
                   alpha=0.6, color='green', s=50)
        max_val = max(max(stats['improvements']['base_entropies']), 
                     max(stats['improvements']['enhanced_entropies']))
        ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        ax2.set_xlabel('Base Average Entropy')
        ax2.set_ylabel('Enhanced Average Entropy')
        ax2.set_title('Improvement Cases: Entropy Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # 3. Improvement Cases - Summary Bar Chart
    ax3 = axes[0, 2]
    if 'improvements' in stats:
        categories = ['Avg PPL', 'Avg Entropy']
        base_vals = [stats['improvements']['base_avg_ppl'], 
                    stats['improvements']['base_avg_entropy']]
        enh_vals = [stats['improvements']['enhanced_avg_ppl'], 
                   stats['improvements']['enhanced_avg_entropy']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, base_vals, width, label='Base', alpha=0.7, color='lightblue')
        bars2 = ax3.bar(x + width/2, enh_vals, width, label='Enhanced', alpha=0.7, color='lightgreen')
        
        ax3.set_ylabel('Value')
        ax3.set_title(f'Improvement Cases Summary (n={stats["improvements"]["count"]})')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Regression Cases - Perplexity Scatter
    ax4 = axes[1, 0]
    if 'regressions' in stats:
        ax4.scatter(stats['regressions']['base_ppls'], 
                   stats['regressions']['enhanced_ppls'], 
                   alpha=0.6, color='red', s=50)
        max_val = max(max(stats['regressions']['base_ppls']), 
                     max(stats['regressions']['enhanced_ppls']))
        ax4.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        ax4.set_xlabel('Base Average Perplexity')
        ax4.set_ylabel('Enhanced Average Perplexity')
        ax4.set_title('Regression Cases: Perplexity Comparison')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # 5. Regression Cases - Entropy Scatter
    ax5 = axes[1, 1]
    if 'regressions' in stats:
        ax5.scatter(stats['regressions']['base_entropies'], 
                   stats['regressions']['enhanced_entropies'], 
                   alpha=0.6, color='red', s=50)
        max_val = max(max(stats['regressions']['base_entropies']), 
                     max(stats['regressions']['enhanced_entropies']))
        ax5.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        ax5.set_xlabel('Base Average Entropy')
        ax5.set_ylabel('Enhanced Average Entropy')
        ax5.set_title('Regression Cases: Entropy Comparison')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    
    # 6. Regression Cases - Summary Bar Chart
    ax6 = axes[1, 2]
    if 'regressions' in stats:
        categories = ['Avg PPL', 'Avg Entropy']
        base_vals = [stats['regressions']['base_avg_ppl'], 
                    stats['regressions']['base_avg_entropy']]
        enh_vals = [stats['regressions']['enhanced_avg_ppl'], 
                   stats['regressions']['enhanced_avg_entropy']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, base_vals, width, label='Base', alpha=0.7, color='lightblue')
        bars2 = ax6.bar(x + width/2, enh_vals, width, label='Enhanced', alpha=0.7, color='lightcoral')
        
        ax6.set_ylabel('Value')
        ax6.set_title(f'Regression Cases Summary (n={stats["regressions"]["count"]})')
        ax6.set_xticks(x)
        ax6.set_xticklabels(categories)
        ax6.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Token Analysis: Base vs Enhanced Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'token_analysis_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    
    # Create additional token position plots
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
    
    # PPL by token position for improvements
    ax1 = axes2[0, 0]
    if 'improvements' in stats and 'improvements' in results:
        # Calculate average PPL for each token position across all samples
        max_len = min(n_tokens, max([len(r['ppls']) for r in results['improvements']['base']] + 
                                   [len(r['ppls']) for r in results['improvements']['enhanced']]))
        
        base_ppl_by_pos = []
        enh_ppl_by_pos = []
        
        for pos in range(max_len):
            base_ppls_at_pos = [r['ppls'][pos] for r in results['improvements']['base'] 
                               if len(r['ppls']) > pos]
            enh_ppls_at_pos = [r['ppls'][pos] for r in results['improvements']['enhanced'] 
                              if len(r['ppls']) > pos]
            
            if base_ppls_at_pos:
                base_ppl_by_pos.append(np.mean(base_ppls_at_pos))
            if enh_ppls_at_pos:
                enh_ppl_by_pos.append(np.mean(enh_ppls_at_pos))
        
        x_positions = range(len(base_ppl_by_pos))
        ax1.plot(x_positions, base_ppl_by_pos, label='Base', color='blue', alpha=0.7, linewidth=2)
        ax1.plot(x_positions[:len(enh_ppl_by_pos)], enh_ppl_by_pos, label='Enhanced', 
                color='green', alpha=0.7, linewidth=2)
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Average Perplexity (log scale)')
        ax1.set_yscale('log')
        ax1.set_title('Improvement Cases: PPL by Token Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3, which='both')
    
    # Entropy by token position for improvements
    ax2 = axes2[0, 1]
    if 'improvements' in stats and 'improvements' in results and results['improvements']['base']:
        max_len = min(n_tokens, max([len(r['entropies']) for r in results['improvements']['base']] + 
                                   [len(r['entropies']) for r in results['improvements']['enhanced']]))
        
        base_ent_by_pos = []
        enh_ent_by_pos = []
        
        for pos in range(max_len):
            base_ents_at_pos = [r['entropies'][pos] for r in results['improvements']['base'] 
                               if len(r['entropies']) > pos]
            enh_ents_at_pos = [r['entropies'][pos] for r in results['improvements']['enhanced'] 
                              if len(r['entropies']) > pos]
            
            if base_ents_at_pos:
                base_ent_by_pos.append(np.mean(base_ents_at_pos))
            if enh_ents_at_pos:
                enh_ent_by_pos.append(np.mean(enh_ents_at_pos))
        
        
        if base_ent_by_pos and enh_ent_by_pos:
            x_positions = range(len(base_ent_by_pos))
            ax2.plot(x_positions, base_ent_by_pos, label='Base', color='blue', alpha=0.7, linewidth=2)
            ax2.plot(x_positions[:len(enh_ent_by_pos)], enh_ent_by_pos, label='Enhanced', 
                    color='green', alpha=0.7, linewidth=2)
            ax2.set_xlabel('Token Position')
            ax2.set_ylabel('Average Entropy')
            ax2.set_title('Improvement Cases: Entropy by Token Position')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No entropy data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Improvement Cases: Entropy by Token Position (No Data)')
    
    # PPL by token position for regressions
    ax3 = axes2[1, 0]
    if 'regressions' in stats and 'regressions' in results and results['regressions']['base']:
        max_len_reg = min(n_tokens, max([len(r['ppls']) for r in results['regressions']['base']] + 
                                       [len(r['ppls']) for r in results['regressions']['enhanced']]))
        
        base_ppl_by_pos_reg = []
        enh_ppl_by_pos_reg = []
        
        for pos in range(max_len_reg):
            base_ppls_at_pos = [r['ppls'][pos] for r in results['regressions']['base'] 
                               if len(r['ppls']) > pos]
            enh_ppls_at_pos = [r['ppls'][pos] for r in results['regressions']['enhanced'] 
                              if len(r['ppls']) > pos]
            
            if base_ppls_at_pos:
                base_ppl_by_pos_reg.append(np.mean(base_ppls_at_pos))
            if enh_ppls_at_pos:
                enh_ppl_by_pos_reg.append(np.mean(enh_ppls_at_pos))
        
        x_positions = range(len(base_ppl_by_pos_reg))
        ax3.plot(x_positions, base_ppl_by_pos_reg, label='Base', color='blue', alpha=0.7, linewidth=2)
        ax3.plot(x_positions[:len(enh_ppl_by_pos_reg)], enh_ppl_by_pos_reg, label='Enhanced', 
                color='red', alpha=0.7, linewidth=2)
        ax3.set_xlabel('Token Position')
        ax3.set_ylabel('Average Perplexity (log scale)')
        ax3.set_yscale('log')
        ax3.set_title('Regression Cases: PPL by Token Position')
        ax3.legend()
        ax3.grid(True, alpha=0.3, which='both')
    
    # Entropy by token position for regressions
    ax4 = axes2[1, 1]
    if 'regressions' in stats and 'regressions' in results and results['regressions']['base']:
        max_len_reg = min(n_tokens, max([len(r['entropies']) for r in results['regressions']['base']] + 
                                       [len(r['entropies']) for r in results['regressions']['enhanced']]))
        
        base_ent_by_pos_reg = []
        enh_ent_by_pos_reg = []
        
        for pos in range(max_len_reg):
            base_ents_at_pos = [r['entropies'][pos] for r in results['regressions']['base'] 
                               if len(r['entropies']) > pos]
            enh_ents_at_pos = [r['entropies'][pos] for r in results['regressions']['enhanced'] 
                              if len(r['entropies']) > pos]
            
            if base_ents_at_pos:
                base_ent_by_pos_reg.append(np.mean(base_ents_at_pos))
            if enh_ents_at_pos:
                enh_ent_by_pos_reg.append(np.mean(enh_ents_at_pos))
        
        if base_ent_by_pos_reg and enh_ent_by_pos_reg:
            x_positions = range(len(base_ent_by_pos_reg))
            ax4.plot(x_positions, base_ent_by_pos_reg, label='Base', color='blue', alpha=0.7, linewidth=2)
            ax4.plot(x_positions[:len(enh_ent_by_pos_reg)], enh_ent_by_pos_reg, label='Enhanced', 
                    color='red', alpha=0.7, linewidth=2)
            ax4.set_xlabel('Token Position')
            ax4.set_ylabel('Average Entropy')
            ax4.set_title('Regression Cases: Entropy by Token Position')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No entropy data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Regression Cases: Entropy by Token Position (No Data)')
    
    plt.suptitle(f'Token Metrics by Position Analysis (First {n_tokens} Tokens)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path2 = os.path.join(output_dir, 'token_metrics_distributions.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    logger.info(f"Token position plots saved to {output_path2}")


def main():
    parser = argparse.ArgumentParser(description='Complete token analysis for base and enhanced models')
    # Get project root directory (parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser.add_argument('--base', default=os.path.join(project_root, 'saves/base.jsonl'), 
                       help='Path to base.jsonl file')
    parser.add_argument('--enhanced', default=os.path.join(project_root, 'saves/enhanced.jsonl'), 
                       help='Path to enhanced.jsonl file')
    parser.add_argument('--output-dir', default=os.path.join(project_root, 'saves/token_analysis_output'), help='Output directory')
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                       help='Model to use for token analysis')
    parser.add_argument('--n-tokens', type=int, default=100, help='Number of tokens to analyze')
    parser.add_argument('--sample-size', type=int, help='Sample size for analysis (optional)')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip token analysis, only save samples')
    parser.add_argument('--include-problem', action='store_true', default=True, 
                       help='Include problem text in token analysis (default: True)')
    parser.add_argument('--response-only', action='store_true', 
                       help='Analyze only response text, excluding problem')
    parser.add_argument('--entropy-method', choices=['full', 'top-k'], default='full',
                       help='Entropy calculation method: full (all tokens) or top-k (top-k tokens only)')
    
    args = parser.parse_args()
    
    # Load and match data
    base_data, enhanced_data, improvements, regressions = load_and_match_data(
        args.base, args.enhanced
    )
    
    # Save samples
    save_samples(base_data, enhanced_data, improvements, regressions, args.output_dir)
    
    if not args.skip_analysis:
        # Initialize analyzer
        analyzer = TokenAnalyzer(args.model)
        
        # Determine if we should include problem text
        include_problem = not args.response_only if args.response_only else args.include_problem
        
        logger.info(f"Token analysis mode: {'Problem + Response' if include_problem else 'Response only'}")
        
        # Analyze samples
        results = analyze_samples(
            analyzer, base_data, enhanced_data, 
            improvements, regressions, 
            args.sample_size, args.n_tokens, include_problem, args.entropy_method
        )
        
        # Calculate statistics
        stats = calculate_statistics(results)
        
        
        # Print summary
        logger.info("\n=== Summary Statistics ===")
        if 'improvements' in stats:
            logger.info(f"\nImprovement Cases (n={stats['improvements']['count']}):")
            logger.info(f"  Base avg PPL: {stats['improvements']['base_avg_ppl']:.2f}")
            logger.info(f"  Enhanced avg PPL: {stats['improvements']['enhanced_avg_ppl']:.2f}")
            logger.info(f"  PPL Change: {stats['improvements']['ppl_change']:.2f}")
            logger.info(f"  Base avg Entropy: {stats['improvements']['base_avg_entropy']:.2f}")
            logger.info(f"  Enhanced avg Entropy: {stats['improvements']['enhanced_avg_entropy']:.2f}")
            logger.info(f"  Entropy Change: {stats['improvements']['entropy_change']:.2f}")
        
        if 'regressions' in stats:
            logger.info(f"\nRegression Cases (n={stats['regressions']['count']}):")
            logger.info(f"  Base avg PPL: {stats['regressions']['base_avg_ppl']:.2f}")
            logger.info(f"  Enhanced avg PPL: {stats['regressions']['enhanced_avg_ppl']:.2f}")
            logger.info(f"  PPL Change: {stats['regressions']['ppl_change']:.2f}")
            logger.info(f"  Base avg Entropy: {stats['regressions']['base_avg_entropy']:.2f}")
            logger.info(f"  Enhanced avg Entropy: {stats['regressions']['enhanced_avg_entropy']:.2f}")
            logger.info(f"  Entropy Change: {stats['regressions']['entropy_change']:.2f}")
        
        # Create visualizations
        create_visualizations(stats, results, args.output_dir, args.n_tokens)
        
        # Save results
        results_path = os.path.join(args.output_dir, 'analysis_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy arrays and numpy scalars to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                else:
                    return obj
            
            stats_json = convert_numpy(stats)
            json.dump(stats_json, f, indent=2)
        logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()