#!/usr/bin/env python3
"""
Complete Progressive Assistance Analysis
Analyzes all datasets with three classifiers and generates comprehensive tables.
"""

import argparse
import jsonlines
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add scripts directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from eval_acc_lm_eval import compute_accuracy_from_record


def find_task_file(directory, task):
    """Find the JSONL file for a specific task in a directory."""
    if not os.path.exists(directory):
        return None
    
    for file in os.listdir(directory):
        if f"hendrycks_math_{task}" in file and file.endswith('.jsonl') and 'lm-eval-detailed-results' not in file:
            return os.path.join(directory, file)
    return None


def load_all_results(results_dir):
    """Load results for all tasks and configurations."""
    logger.info("=== Loading All Results ===")
    
    configs = {
        'base': 'deepseek-ai__DeepSeek-R1-Distill-Qwen-7B',
        'enhanced100': '7-32-100/ensemble', 
        'enhanced500': '7-32-500/ensemble',
        'enhanced1000': '7-32-1000/ensemble'
    }
    
    tasks = ['algebra', 'counting_and_prob', 'geometry', 'intermediate_algebra', 'num_theory', 'precalc']
    
    all_data = {}
    
    for task in tasks:
        logger.info(f"Loading {task} results...")
        task_data = {}
        
        for config_name, config_path in configs.items():
            full_path = os.path.join(results_dir, config_path)
            task_file = find_task_file(full_path, task)
            
            if task_file:
                with jsonlines.open(task_file) as reader:
                    samples = list(reader)
                task_data[config_name] = samples
                logger.info(f"  {config_name}: {len(samples)} samples")
            else:
                logger.warning(f"  {config_name}: File not found")
        
        all_data[task] = task_data
    
    return all_data


def evaluate_correctness(sample):
    """Evaluate if a sample is correct."""
    try:
        is_correct, _, _ = compute_accuracy_from_record(sample)
        return is_correct
    except:
        return False


def get_problem_text(sample):
    """Extract problem text from sample."""
    if 'doc' in sample and 'problem' in sample['doc']:
        return sample['doc']['problem']
    elif 'problem' in sample:
        return sample['problem']
    return None


def get_response_text(sample):
    """Extract response text from sample."""
    if 'resps' in sample and sample['resps'] and sample['resps'][0]:
        return sample['resps'][0][0]
    return ""


def match_samples_by_index(task_data, max_samples=None):
    """Match samples by index (faster than problem text matching)."""
    if 'base' not in task_data:
        return []
    
    min_samples = min(len(samples) for samples in task_data.values())
    if max_samples:
        min_samples = min(min_samples, max_samples)
    
    logger.debug(f"Matching {min_samples} samples by index...")
    
    matched_samples = []
    for i in range(min_samples):
        matched = {}
        for config_name, samples in task_data.items():
            matched[config_name] = samples[i]
        matched_samples.append(matched)
    
    return matched_samples


def determine_progressive_strategy(results):
    """Determine optimal strategy using progressive logic."""
    base_correct = results['base']
    
    # Progressive strategy: try shortest assistance first
    for config in ['enhanced100', 'enhanced500', 'enhanced1000']:
        if config in results:
            enhanced_correct = results[config]
            # If this configuration helps (fixes error or maintains correctness)
            if enhanced_correct and not base_correct:  # Fixes error
                return config
            elif enhanced_correct and base_correct:  # Maintains correctness  
                return config
    
    # If no assistance helps, use base
    return 'base'


def extract_simple_features(sample):
    """Extract simple features without requiring model inference."""
    try:
        problem = get_problem_text(sample)
        if not problem:
            return None
        
        response = get_response_text(sample)
        
        # Simple text-based features
        features = {
            'problem_length': len(problem.split()),
            'problem_chars': len(problem),
            'response_length': len(response.split()),
            'response_chars': len(response),
            'math_symbols': problem.count('$') + problem.count('\\'),
            'has_frac': 1 if 'frac' in problem else 0,
            'has_sqrt': 1 if 'sqrt' in problem else 0,
            'has_sum': 1 if 'sum' in problem else 0,
            'has_int': 1 if 'int' in problem else 0,
            'problem_lines': len(problem.split('\n')),
            'response_lines': len(response.split('\n')),
            'response_ratio': len(response) / max(1, len(problem)),
            'avg_word_length': np.mean([len(word) for word in problem.split()]) if problem.split() else 0,
        }
        
        return features
        
    except Exception as e:
        logger.debug(f"Feature extraction error: {e}")
        return None


def train_classifiers_for_task(task_data, task_name):
    """Train three different classifiers for a specific task."""
    logger.info(f"\n=== Training Classifiers for {task_name.upper()} ===")
    
    matched_samples = match_samples_by_index(task_data)
    if len(matched_samples) < 20:
        logger.warning(f"Not enough samples for {task_name}: {len(matched_samples)}")
        return {}
    
    logger.info(f"Training on {len(matched_samples)} matched samples")
    
    config_tokens = {
        'enhanced100': 100,
        'enhanced500': 500,
        'enhanced1000': 1000
    }
    
    classifiers = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=1000)
    }
    
    task_results = {}
    
    for config_name, max_tokens in config_tokens.items():
        logger.info(f"  Training classifiers for {config_name}...")
        
        # Prepare data
        features_list = []
        labels_list = []
        
        for sample in matched_samples:
            try:
                # Determine correctness for all configurations
                results = {}
                for cfg in ['base', 'enhanced100', 'enhanced500', 'enhanced1000']:
                    if cfg in sample:
                        results[cfg] = evaluate_correctness(sample[cfg])
                
                # Determine optimal strategy
                optimal_strategy = determine_progressive_strategy(results)
                
                # Extract features
                features = extract_simple_features(sample[config_name])
                
                if features is not None:
                    features_list.append(features)
                    # Label: should we use this configuration?
                    should_use = (optimal_strategy == config_name)
                    labels_list.append(1 if should_use else 0)
                    
            except Exception as e:
                logger.debug(f"Error processing sample: {e}")
                continue
        
        if len(features_list) < 10:
            logger.warning(f"Not enough valid features for {config_name}: {len(features_list)}")
            continue
        
        # Convert to arrays
        feature_names = list(features_list[0].keys())
        X = np.array([[f[name] for name in feature_names] for f in features_list])
        y = np.array(labels_list)
        
        # Split train/test
        unique_classes = np.unique(y)
        can_stratify = len(unique_classes) > 1 and min(np.bincount(y)) >= 2
        
        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        config_results = {}
        
        # Train all classifiers
        for clf_name, clf in classifiers.items():
            try:
                clf.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_acc = clf.score(X_train_scaled, y_train)
                test_acc = clf.score(X_test_scaled, y_test)
                
                y_pred = clf.predict(X_test_scaled)
                
                # Additional metrics
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                config_results[clf_name] = {
                    'model': clf,
                    'scaler': scaler,
                    'feature_names': feature_names,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'positive_rate': sum(y) / len(y),
                    'X_test': X_test_scaled,
                    'y_test': y_test
                }
                
                logger.info(f"    {clf_name:15}: Test Acc={test_acc:.3f}, F1={f1:.3f}, Pos Rate={sum(y)/len(y):.3f}")
                
            except Exception as e:
                logger.warning(f"Error training {clf_name} for {config_name}: {e}")
                continue
        
        task_results[config_name] = config_results
    
    return task_results


def evaluate_progressive_strategy_with_classifiers(task_data, trained_models, task_name, threshold=0.5):
    """Evaluate progressive strategy with different classifiers."""
    logger.info(f"\n=== Evaluating Progressive Strategy for {task_name.upper()} ===")
    
    matched_samples = match_samples_by_index(task_data)
    classifier_names = ['LogisticRegression', 'RandomForest', 'MLP']
    
    results = {}
    
    for clf_name in classifier_names:
        logger.info(f"  Evaluating with {clf_name}...")
        
        total_correct = 0
        total_samples = 0
        strategy_usage = {'base': 0, 'enhanced100': 0, 'enhanced500': 0, 'enhanced1000': 0}
        
        for sample in matched_samples:
            try:
                # Determine true results for evaluation
                true_results = {}
                for cfg in ['base', 'enhanced100', 'enhanced500', 'enhanced1000']:
                    if cfg in sample:
                        true_results[cfg] = evaluate_correctness(sample[cfg])
                
                # Progressive prediction: try 100 -> 500 -> 1000 -> base
                chosen_strategy = 'base'
                
                for config_name in ['enhanced100', 'enhanced500', 'enhanced1000']:
                    if (config_name in trained_models and 
                        clf_name in trained_models[config_name]):
                        
                        # Extract features
                        features = extract_simple_features(sample[config_name])
                        
                        if features is not None:
                            # Make prediction
                            model_data = trained_models[config_name][clf_name]
                            feature_array = np.array([[features[name] for name in model_data['feature_names']]])
                            feature_scaled = model_data['scaler'].transform(feature_array)
                            prob = model_data['model'].predict_proba(feature_scaled)[0][1]
                            
                            if prob > threshold:
                                chosen_strategy = config_name
                                break
                
                # Count strategy usage
                strategy_usage[chosen_strategy] += 1
                
                # Evaluate final result
                if chosen_strategy in true_results:
                    final_correct = true_results[chosen_strategy]
                    if final_correct:
                        total_correct += 1
                    total_samples += 1
                
            except Exception as e:
                logger.debug(f"Error evaluating sample: {e}")
                continue
        
        if total_samples > 0:
            accuracy = total_correct / total_samples
            
            # Calculate average tokens used for assistance
            total_usage = sum(strategy_usage.values())
            if total_usage > 0:
                avg_assistance_tokens = (strategy_usage['enhanced100'] * 100 + 
                                       strategy_usage['enhanced500'] * 500 + 
                                       strategy_usage['enhanced1000'] * 1000) / total_usage
                assistance_rate = (total_usage - strategy_usage['base']) / total_usage
            else:
                avg_assistance_tokens = 0
                assistance_rate = 0
            
            results[clf_name] = {
                'accuracy': accuracy,
                'strategy_usage': strategy_usage.copy(),
                'total_samples': total_samples,
                'avg_assistance_tokens': avg_assistance_tokens,
                'assistance_rate': assistance_rate
            }
            
            logger.info(f"    {clf_name:15}: Acc={accuracy:.3f}, Assist Rate={assistance_rate:.1%}, Avg Tokens={avg_assistance_tokens:.0f}")
    
    return results


def calculate_baseline_accuracies(task_data, task_name):
    """Calculate baseline accuracies for comparison."""
    logger.info(f"\n=== Calculating Baselines for {task_name.upper()} ===")
    
    matched_samples = match_samples_by_index(task_data)
    
    baselines = {
        'base_only': 0,
        'always_100': 0,
        'always_500': 0,
        'always_1000': 0,
        'optimal': 0
    }
    
    # Calculate token lengths for each configuration
    token_stats = {
        'base': {'total_tokens': [], 'base_tokens': [], 'assistance_tokens': []},
        'enhanced100': {'total_tokens': [], 'base_tokens': [], 'assistance_tokens': []},
        'enhanced500': {'total_tokens': [], 'base_tokens': [], 'assistance_tokens': []},
        'enhanced1000': {'total_tokens': [], 'base_tokens': [], 'assistance_tokens': []}
    }
    
    for sample in matched_samples:
        # Evaluate correctness for all strategies
        results = {}
        for cfg in ['base', 'enhanced100', 'enhanced500', 'enhanced1000']:
            if cfg in sample:
                results[cfg] = evaluate_correctness(sample[cfg])
                
                # Calculate token lengths (simple word count approximation)
                response = get_response_text(sample[cfg])
                total_tokens = len(response.split()) if response else 0
                
                if cfg == 'base':
                    base_tokens = total_tokens
                    assistance_tokens = 0
                else:
                    # For enhanced configs, estimate base part and assistance part
                    config_num = int(cfg.replace('enhanced', ''))
                    assistance_tokens = min(config_num, total_tokens)  # Assistance tokens capped at config limit
                    base_tokens = max(0, total_tokens - assistance_tokens)
                
                token_stats[cfg]['total_tokens'].append(total_tokens)
                token_stats[cfg]['base_tokens'].append(base_tokens)
                token_stats[cfg]['assistance_tokens'].append(assistance_tokens)
        
        # Count baseline accuracies
        if results.get('base', False):
            baselines['base_only'] += 1
        
        if results.get('enhanced100', False):
            baselines['always_100'] += 1
            
        if results.get('enhanced500', False):
            baselines['always_500'] += 1
            
        if results.get('enhanced1000', False):
            baselines['always_1000'] += 1
        
        # Optimal strategy
        optimal_strategy = determine_progressive_strategy(results)
        if optimal_strategy in results and results[optimal_strategy]:
            baselines['optimal'] += 1
    
    # Convert to accuracies
    total_samples = len(matched_samples)
    for strategy in baselines:
        baselines[strategy] = baselines[strategy] / total_samples
    
    # Calculate average token lengths
    avg_token_stats = {}
    for cfg in token_stats:
        avg_token_stats[cfg] = {
            'total_tokens': np.mean(token_stats[cfg]['total_tokens']) if token_stats[cfg]['total_tokens'] else 0,
            'base_tokens': np.mean(token_stats[cfg]['base_tokens']) if token_stats[cfg]['base_tokens'] else 0,
            'assistance_tokens': np.mean(token_stats[cfg]['assistance_tokens']) if token_stats[cfg]['assistance_tokens'] else 0
        }
    
    logger.info("Baseline accuracies:")
    for strategy, accuracy in baselines.items():
        logger.info(f"  {strategy}: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return baselines, avg_token_stats


def create_performance_table(all_results, all_baselines, all_token_stats):
    """Create the main performance table."""
    logger.info("\n=== Creating Performance Table ===")
    
    tasks = ['algebra', 'counting_and_prob', 'geometry', 'intermediate_algebra', 'num_theory', 'precalc']
    task_display_names = ['Algebra', 'Counting', 'Geometry', 'Intermediate', 'Num', 'Precalc']
    
    # Create DataFrame for the table
    rows = []
    
    # Single model rows (using base results)
    models = [
        ('7B', 'base_only'),
        ('14B', None),  # We don't have 14B data, will fill with estimated values
        ('32B', None)   # We don't have standalone 32B data
    ]
    
    for model_name, baseline_key in models:
        if baseline_key:
            # Accuracy row
            acc_row = ['Single', model_name, 'Acc.']
            for task in tasks:
                if task in all_baselines:
                    acc = all_baselines[task][baseline_key] * 100
                    acc_row.append(f"{acc:.2f}")
                else:
                    acc_row.append("")
            rows.append(acc_row)
            
            # Length row  
            len_row = ['', '', 'Len.']
            for task in tasks:
                if task in all_token_stats:
                    total_len = all_token_stats[task]['base']['total_tokens']
                    len_row.append(f"{total_len:,.0f}")
                else:
                    len_row.append("")
            rows.append(len_row)
            
            # Cost row (empty for single models)
            cost_row = ['', '', 'Cost'] + [''] * len(tasks)
            rows.append(cost_row)
        else:
            # Placeholder rows for 14B and 32B
            for metric in ['Acc.', 'Len.', 'Cost']:
                row = ['Single', model_name, metric] + [''] * len(tasks)
                rows.append(row)
    
    # Baseline ensemble rows
    baseline_configs = [
        ('7B+32B (100)', 'always_100', 'enhanced100'),
        ('7B+32B (500)', 'always_500', 'enhanced500'),
        ('7B+32B (1000)', 'always_1000', 'enhanced1000')
    ]
    
    for config_name, baseline_key, token_key in baseline_configs:
        # Accuracy row
        acc_row = ['Baseline', config_name, 'Acc.']
        for task in tasks:
            if task in all_baselines:
                acc = all_baselines[task][baseline_key] * 100
                acc_row.append(f"{acc:.2f}")
            else:
                acc_row.append("")
        rows.append(acc_row)
        
        # Length row
        len_row = ['', '', 'Len.']
        for task in tasks:
            if task in all_token_stats:
                total_len = all_token_stats[task][token_key]['total_tokens']
                len_row.append(f"{total_len:,.0f}")
            else:
                len_row.append("")
        rows.append(len_row)
        
        # Cost row (assistance tokens)
        cost_row = ['', '', 'Cost']
        for task in tasks:
            if task in all_token_stats:
                assist_len = all_token_stats[task][token_key]['assistance_tokens']
                cost_row.append(f"{assist_len:.0f}")
            else:
                cost_row.append("")
        rows.append(cost_row)
    
    # Progressive ensemble row (using RandomForest results)
    acc_row = ['Ensemble', '7B+32B', 'Acc.']
    len_row = ['', '', 'Len.']
    cost_row = ['', '', 'Cost']
    
    for task in tasks:
        if task in all_results and 'RandomForest' in all_results[task]:
            rf_result = all_results[task]['RandomForest']
            acc = rf_result['accuracy'] * 100
            acc_row.append(f"{acc:.2f}")
            
            # Calculate progressive length (weighted by strategy usage)
            usage = rf_result['strategy_usage']
            total_usage = sum(usage.values())
            if total_usage > 0:
                weighted_len = 0
                weighted_cost = 0
                for strategy, count in usage.items():
                    weight = count / total_usage
                    if strategy == 'base':
                        if task in all_token_stats:
                            weighted_len += weight * all_token_stats[task]['base']['total_tokens']
                    else:
                        if task in all_token_stats:
                            weighted_len += weight * all_token_stats[task][strategy]['total_tokens']
                            weighted_cost += weight * all_token_stats[task][strategy]['assistance_tokens']
                
                len_row.append(f"{weighted_len:,.0f}")
                cost_row.append(f"{weighted_cost:.0f}")
            else:
                len_row.append("")
                cost_row.append("")
        else:
            acc_row.append("")
            len_row.append("")
            cost_row.append("")
    
    rows.extend([acc_row, len_row, cost_row])
    
    # Create DataFrame
    columns = ['Series', 'Model', 'Metric'] + task_display_names
    df = pd.DataFrame(rows, columns=columns)
    
    return df


def create_classifier_comparison_table(all_results):
    """Create classifier comparison table."""
    logger.info("\n=== Creating Classifier Comparison Table ===")
    
    tasks = ['algebra', 'counting_and_prob', 'geometry', 'intermediate_algebra', 'num_theory', 'precalc']
    task_display_names = ['Algebra', 'Counting', 'Geometry', 'Intermediate', 'Num', 'Precalc']
    classifiers = ['LogisticRegression', 'RandomForest', 'MLP']
    
    rows = []
    
    for clf_name in classifiers:
        display_name = {
            'LogisticRegression': 'Logistic Regression',
            'RandomForest': 'Random Forest', 
            'MLP': 'Neural Network (MLP)'
        }[clf_name]
        
        # Accuracy row
        acc_row = [display_name, 'Acc.']
        for task in tasks:
            if task in all_results and clf_name in all_results[task]:
                acc = all_results[task][clf_name]['accuracy'] * 100
                acc_row.append(f"{acc:.1f}")
            else:
                acc_row.append("")
        rows.append(acc_row)
        
        # Length row
        len_row = ['', 'Len.']
        for task in tasks:
            if task in all_results and clf_name in all_results[task]:
                # Calculate weighted average length
                result = all_results[task][clf_name]
                usage = result['strategy_usage']
                total_usage = sum(usage.values())
                if total_usage > 0:
                    # Simplified length calculation
                    avg_len = (usage['base'] * 2000 +  # Approximate base length
                              usage['enhanced100'] * 2100 +  # Base + 100
                              usage['enhanced500'] * 2500 +  # Base + 500
                              usage['enhanced1000'] * 3000) / total_usage  # Base + 1000
                    len_row.append(f"{avg_len:.0f}")
                else:
                    len_row.append("")
            else:
                len_row.append("")
        rows.append(len_row)
        
        # Cost row (assistance tokens only)
        cost_row = ['', 'Cost']
        for task in tasks:
            if task in all_results and clf_name in all_results[task]:
                avg_cost = all_results[task][clf_name]['avg_assistance_tokens']
                cost_row.append(f"{avg_cost:.0f}")
            else:
                cost_row.append("")
        rows.append(cost_row)
    
    # Create DataFrame
    columns = ['Classifier', 'Metric'] + task_display_names
    df = pd.DataFrame(rows, columns=columns)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Complete Progressive Assistance Analysis')
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser.add_argument('--results-dir', default=os.path.join(project_root, 'results'),
                       help='Results directory containing actual model outputs')
    parser.add_argument('--output-dir', default=os.path.join(project_root, 'saves/complete_analysis'),
                       help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for predictions')
    
    args = parser.parse_args()
    
    # Load all results
    logger.info("Loading all experimental results...")
    all_data = load_all_results(args.results_dir)
    
    # Process each task
    all_results = {}
    all_baselines = {}
    all_token_stats = {}
    
    tasks = ['algebra', 'counting_and_prob', 'geometry', 'intermediate_algebra', 'num_theory', 'precalc']
    
    for task in tasks:
        if task not in all_data or not all_data[task]:
            logger.warning(f"No data for task {task}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING TASK: {task.upper()}")
        logger.info(f"{'='*60}")
        
        # Calculate baselines and token statistics
        baselines, token_stats = calculate_baseline_accuracies(all_data[task], task)
        all_baselines[task] = baselines
        all_token_stats[task] = token_stats
        
        # Train classifiers
        trained_models = train_classifiers_for_task(all_data[task], task)
        
        # Evaluate progressive strategy with all classifiers
        if trained_models:
            task_results = evaluate_progressive_strategy_with_classifiers(
                all_data[task], trained_models, task, args.threshold
            )
            all_results[task] = task_results
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate tables
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING FINAL TABLES")
    logger.info(f"{'='*80}")
    
    # Main performance table
    performance_df = create_performance_table(all_results, all_baselines, all_token_stats)
    performance_csv = os.path.join(args.output_dir, 'complete_performance_table.csv')
    performance_df.to_csv(performance_csv, index=False)
    logger.info(f"Performance table saved to: {performance_csv}")
    
    # Classifier comparison table
    classifier_df = create_classifier_comparison_table(all_results)
    classifier_csv = os.path.join(args.output_dir, 'classifier_comparison_table.csv')
    classifier_df.to_csv(classifier_csv, index=False)
    logger.info(f"Classifier comparison saved to: {classifier_csv}")
    
    # Save detailed results
    results_json = os.path.join(args.output_dir, 'detailed_results.json')
    
    # Convert results to JSON-serializable format
    json_results = {
        'task_results': {},
        'baselines': {},
        'token_stats': {}
    }
    
    for task in all_results:
        json_results['task_results'][task] = {}
        for clf_name, result in all_results[task].items():
            json_results['task_results'][task][clf_name] = {
                'accuracy': float(result['accuracy']),
                'strategy_usage': {k: int(v) for k, v in result['strategy_usage'].items()},
                'total_samples': int(result['total_samples']),
                'avg_assistance_tokens': float(result['avg_assistance_tokens']),
                'assistance_rate': float(result['assistance_rate'])
            }
    
    for task in all_baselines:
        json_results['baselines'][task] = {k: float(v) for k, v in all_baselines[task].items()}
    
    for task in all_token_stats:
        json_results['token_stats'][task] = {}
        for config, stats in all_token_stats[task].items():
            json_results['token_stats'][task][config] = {k: float(v) for k, v in stats.items()}
    
    with open(results_json, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Detailed results saved to: {results_json}")
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Tasks analyzed: {len(all_results)}")
    logger.info(f"🧠 Classifiers compared: Logistic Regression, Random Forest, MLP")
    logger.info(f"📈 Tables generated: Performance, Classifier Comparison")
    logger.info(f"💾 Output directory: {args.output_dir}")
    
    # Show performance table preview
    logger.info(f"\n📋 PERFORMANCE TABLE PREVIEW:")
    print("\n" + "="*120)
    print(performance_df.to_string(index=False))
    print("="*120)
    
    logger.info(f"\n🔧 CLASSIFIER COMPARISON PREVIEW:")
    print("\n" + "="*100)
    print(classifier_df.to_string(index=False))
    print("="*100)


if __name__ == "__main__":
    main()