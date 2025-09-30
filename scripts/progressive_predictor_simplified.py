#!/usr/bin/env python3
"""
Simplified Progressive Assistance Predictor
Implements the recursive logic: try 100 -> 500 -> 1000 -> base only
Uses actual result files from experiments.
"""

import argparse
import jsonlines
import json
import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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

from complete_token_analysis import TokenAnalyzer
from eval_acc_lm_eval import compute_accuracy_from_record


def find_task_file(directory, task):
    """Find the JSONL file for a specific task in a directory."""
    if not os.path.exists(directory):
        return None
    
    for file in os.listdir(directory):
        if f"hendrycks_math_{task}" in file and file.endswith('.jsonl') and 'lm-eval-detailed-results' not in file:
            return os.path.join(directory, file)
    return None


def load_actual_results(results_dir):
    """Load actual results from the results directory."""
    logger.info("=== Loading Actual Results ===")
    
    # Define configurations
    configs = {
        'base': 'deepseek-ai__DeepSeek-R1-Distill-Qwen-7B',
        'enhanced100': '7-32-100/ensemble', 
        'enhanced500': '7-32-500/ensemble',
        'enhanced1000': '7-32-1000/ensemble'
    }
    
    # Define tasks
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


def match_samples_by_problem(task_data):
    """Match samples across configurations by problem text."""
    if 'base' not in task_data:
        return []
    
    base_samples = task_data['base']
    matched_samples = []
    
    for base_sample in base_samples:
        try:
            # Handle different data structures
            if 'doc' in base_sample and 'problem' in base_sample['doc']:
                problem = base_sample['doc']['problem']
            elif 'problem' in base_sample:
                problem = base_sample['problem']
            else:
                continue
            
            # Create matched sample starting with base
            matched = {'base': base_sample}
            
            # Find corresponding samples in other configurations
            all_matched = True
            for config_name in ['enhanced100', 'enhanced500', 'enhanced1000']:
                if config_name not in task_data:
                    all_matched = False
                    break
                    
                found = False
                for config_sample in task_data[config_name]:
                    try:
                        # Handle different data structures
                        if 'doc' in config_sample and 'problem' in config_sample['doc']:
                            config_problem = config_sample['doc']['problem']
                        elif 'problem' in config_sample:
                            config_problem = config_sample['problem']
                        else:
                            continue
                        
                        if config_problem == problem:
                            matched[config_name] = config_sample
                            found = True
                            break
                    except:
                        continue
                
                if not found:
                    all_matched = False
                    break
            
            # Only include if all configurations have this problem
            if all_matched:
                matched_samples.append(matched)
                
        except Exception as e:
            logger.debug(f"Error matching sample: {e}")
            continue
    
    return matched_samples


def evaluate_correctness(sample):
    """Evaluate if a sample is correct."""
    try:
        is_correct, _, _ = compute_accuracy_from_record(sample)
        return is_correct
    except:
        return False


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


def extract_features_with_analyzer(analyzer, problem, response, max_tokens):
    """Extract features using token analyzer."""
    try:
        tokens, ppls, entropies = analyzer.compute_answer_token_metrics(
            problem, response, max_tokens
        )
        
        if not ppls or not entropies or len(ppls) < 10:
            return None
        
        features = {
            'ppl_mean': np.mean(ppls),
            'ppl_std': np.std(ppls),
            'ppl_median': np.median(ppls),
            'ppl_max': np.max(ppls),
            'ppl_min': np.min(ppls),
            'entropy_mean': np.mean(entropies),
            'entropy_std': np.std(entropies),
            'entropy_median': np.median(entropies),
            'entropy_max': np.max(entropies),
            'entropy_min': np.min(entropies),
            'token_count': len(ppls),
            'problem_length': len(problem.split())
        }
        
        # Add trend features if enough tokens
        if len(ppls) >= 20:
            mid_point = len(ppls) // 2
            features['ppl_trend'] = np.mean(ppls[mid_point:]) - np.mean(ppls[:mid_point])
            features['entropy_trend'] = np.mean(entropies[mid_point:]) - np.mean(entropies[:mid_point])
        else:
            features['ppl_trend'] = 0.0
            features['entropy_trend'] = 0.0
        
        return features
        
    except Exception as e:
        logger.debug(f"Feature extraction error: {e}")
        return None


def train_progressive_classifiers(all_data, analyzer):
    """Train classifiers for progressive assistance prediction."""
    logger.info("=== Training Progressive Classifiers ===")
    
    # Configuration mapping: config_name -> max_tokens
    config_tokens = {
        'enhanced100': 100,
        'enhanced500': 500,
        'enhanced1000': 1000
    }
    
    all_results = {}
    
    for task, task_data in all_data.items():
        logger.info(f"\nProcessing {task}...")
        
        # Match samples
        matched_samples = match_samples_by_problem(task_data)
        if len(matched_samples) < 20:
            logger.warning(f"Not enough matched samples for {task}: {len(matched_samples)}")
            continue
        
        logger.info(f"Matched {len(matched_samples)} samples for {task}")
        
        # Prepare data for each configuration
        task_results = {}
        
        for config_name, max_tokens in config_tokens.items():
            logger.info(f"  Training classifier for {config_name}...")
            
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
                    
                    # Extract features for current configuration
                    if 'doc' in sample['base'] and 'problem' in sample['base']['doc']:
                        problem = sample['base']['doc']['problem']
                    elif 'problem' in sample['base']:
                        problem = sample['base']['problem']
                    else:
                        continue
                    response = sample[config_name]['resps'][0][0] if sample[config_name]['resps'] else ""
                    
                    features = extract_features_with_analyzer(analyzer, problem, response, max_tokens)
                    
                    if features is not None:
                        features_list.append(features)
                        # Label: should we use this configuration?
                        should_use = (optimal_strategy == config_name)
                        labels_list.append(1 if should_use else 0)
                        
                except Exception as e:
                    logger.debug(f"Error processing sample: {e}")
                    continue
            
            if len(features_list) < 10:
                logger.warning(f"Not enough valid features for {config_name} in {task}: {len(features_list)}")
                continue
            
            # Convert to arrays
            feature_names = list(features_list[0].keys())
            X = np.array([[f[name] for name in feature_names] for f in features_list])
            y = np.array(labels_list)
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_acc = rf.score(X_train_scaled, y_train)
            test_acc = rf.score(X_test_scaled, y_test)
            
            positive_rate = sum(y) / len(y)
            
            logger.info(f"    Train acc: {train_acc:.3f}, Test acc: {test_acc:.3f}, Positive rate: {positive_rate:.3f}")
            
            task_results[config_name] = {
                'model': rf,
                'scaler': scaler,
                'feature_names': feature_names,
                'test_accuracy': test_acc,
                'positive_rate': positive_rate,
                'X_test': X_test_scaled,
                'y_test': y_test
            }
        
        all_results[task] = task_results
    
    return all_results


def evaluate_progressive_strategy(all_data, trained_models, analyzer, threshold=0.5):
    """Evaluate the progressive strategy on test data."""
    logger.info("=== Evaluating Progressive Strategy ===")
    
    config_tokens = {
        'enhanced100': 100,
        'enhanced500': 500,
        'enhanced1000': 1000
    }
    
    overall_results = {}
    total_correct = 0
    total_samples = 0
    overall_strategy_usage = {'base': 0, 'enhanced100': 0, 'enhanced500': 0, 'enhanced1000': 0}
    
    for task, task_data in all_data.items():
        if task not in trained_models:
            continue
        
        logger.info(f"\nEvaluating {task}...")
        
        # Match samples
        matched_samples = match_samples_by_problem(task_data)
        
        task_correct = 0
        task_total = 0
        task_strategy_usage = {'base': 0, 'enhanced100': 0, 'enhanced500': 0, 'enhanced1000': 0}
        
        for sample in tqdm(matched_samples, desc=f"Processing {task}"):
            try:
                # Determine true results for evaluation
                true_results = {}
                for cfg in ['base', 'enhanced100', 'enhanced500', 'enhanced1000']:
                    if cfg in sample:
                        true_results[cfg] = evaluate_correctness(sample[cfg])
                
                # Progressive prediction: try 100 -> 500 -> 1000 -> base
                if 'doc' in sample['base'] and 'problem' in sample['base']['doc']:
                    problem = sample['base']['doc']['problem']
                elif 'problem' in sample['base']:
                    problem = sample['base']['problem']
                else:
                    continue
                chosen_strategy = 'base'
                
                for config_name in ['enhanced100', 'enhanced500', 'enhanced1000']:
                    if config_name in trained_models[task]:
                        # Extract features
                        response = sample[config_name]['resps'][0][0] if sample[config_name]['resps'] else ""
                        max_tokens = config_tokens[config_name]
                        
                        features = extract_features_with_analyzer(analyzer, problem, response, max_tokens)
                        
                        if features is not None:
                            # Make prediction
                            model_data = trained_models[task][config_name]
                            feature_array = np.array([[features[name] for name in model_data['feature_names']]])
                            feature_scaled = model_data['scaler'].transform(feature_array)
                            prob = model_data['model'].predict_proba(feature_scaled)[0][1]
                            
                            if prob > threshold:
                                chosen_strategy = config_name
                                break
                
                # Count strategy usage
                task_strategy_usage[chosen_strategy] += 1
                
                # Evaluate final result
                if chosen_strategy in true_results:
                    final_correct = true_results[chosen_strategy]
                    if final_correct:
                        task_correct += 1
                    task_total += 1
                
            except Exception as e:
                logger.debug(f"Error evaluating sample: {e}")
                continue
        
        if task_total > 0:
            task_accuracy = task_correct / task_total
            logger.info(f"  Accuracy: {task_accuracy:.3f} ({task_correct}/{task_total})")
            logger.info(f"  Strategy usage: {task_strategy_usage}")
            
            overall_results[task] = {
                'accuracy': task_accuracy,
                'strategy_usage': task_strategy_usage,
                'total_samples': task_total
            }
            
            total_correct += task_correct
            total_samples += task_total
            
            for strategy, count in task_strategy_usage.items():
                overall_strategy_usage[strategy] += count
    
    # Overall summary
    if total_samples > 0:
        overall_accuracy = total_correct / total_samples
        
        logger.info(f"\n=== OVERALL RESULTS ===")
        logger.info(f"Progressive accuracy: {overall_accuracy:.3f} ({total_correct}/{total_samples})")
        logger.info(f"Overall strategy usage: {overall_strategy_usage}")
        
        # Calculate average tokens used for assistance
        total_usage = sum(overall_strategy_usage.values())
        if total_usage > 0:
            avg_tokens = (overall_strategy_usage['enhanced100'] * 100 + 
                         overall_strategy_usage['enhanced500'] * 500 + 
                         overall_strategy_usage['enhanced1000'] * 1000) / total_usage
            logger.info(f"Average assistance tokens: {avg_tokens:.0f}")
            
            assistance_rate = (total_usage - overall_strategy_usage['base']) / total_usage
            logger.info(f"Assistance rate: {assistance_rate:.1%}")
        
        return overall_accuracy, overall_results, overall_strategy_usage
    
    return 0.0, {}, {}


def main():
    parser = argparse.ArgumentParser(description='Simplified Progressive Assistance Predictor')
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser.add_argument('--results-dir', default=os.path.join(project_root, 'results'),
                       help='Results directory containing actual model outputs')
    parser.add_argument('--output-dir', default=os.path.join(project_root, 'saves/progressive_output'),
                       help='Output directory')
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                       help='7B model to use for feature extraction (ppl/entropy)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for predictions')
    
    args = parser.parse_args()
    
    # Load actual results
    logger.info("Loading actual experimental results...")
    all_data = load_actual_results(args.results_dir)
    
    # Initialize analyzer for 7B model (for ppl/entropy computation)
    logger.info(f"Initializing token analyzer with {args.model}...")
    analyzer = TokenAnalyzer(args.model)
    
    # Train progressive classifiers
    trained_models = train_progressive_classifiers(all_data, analyzer)
    
    # Evaluate progressive strategy
    overall_accuracy, task_results, strategy_usage = evaluate_progressive_strategy(
        all_data, trained_models, analyzer, args.threshold
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, 'progressive_results.json')
    
    results = {
        'overall_accuracy': float(overall_accuracy),
        'strategy_usage': {k: int(v) for k, v in strategy_usage.items()},
        'task_results': {}
    }
    
    for task, task_result in task_results.items():
        results['task_results'][task] = {
            'accuracy': float(task_result['accuracy']),
            'strategy_usage': {k: int(v) for k, v in task_result['strategy_usage'].items()},
            'total_samples': int(task_result['total_samples'])
        }
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {result_file}")
    
    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"{'PROGRESSIVE ASSISTANCE PREDICTOR':^60}")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Logic: Try 100 → 500 → 1000 → Base only")
    logger.info(f"🎯 Overall Accuracy: {overall_accuracy:.3f}")
    logger.info(f"📈 Strategy Usage: {strategy_usage}")
    
    if sum(strategy_usage.values()) > 0:
        avg_tokens = (strategy_usage['enhanced100'] * 100 + 
                     strategy_usage['enhanced500'] * 500 + 
                     strategy_usage['enhanced1000'] * 1000) / sum(strategy_usage.values())
        logger.info(f"⚡ Average Tokens: {avg_tokens:.0f}")
    
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()