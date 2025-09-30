#!/usr/bin/env python3
"""
Fast Progressive Assistance Predictor Test
Quick test of progressive logic: 100 → 500 → 1000 → base
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


def load_sample_data(results_dir, max_samples=50):
    """Load a small sample of data for fast testing."""
    logger.info("=== Loading Sample Data ===")
    
    configs = {
        'base': 'deepseek-ai__DeepSeek-R1-Distill-Qwen-7B',
        'enhanced100': '7-32-100/ensemble', 
        'enhanced500': '7-32-500/ensemble',
        'enhanced1000': '7-32-1000/ensemble'
    }
    
    # Test with one task first
    task = 'algebra'
    logger.info(f"Loading {task} results (max {max_samples} samples)...")
    
    task_data = {}
    
    for config_name, config_path in configs.items():
        full_path = os.path.join(results_dir, config_path)
        task_file = find_task_file(full_path, task)
        
        if task_file:
            with jsonlines.open(task_file) as reader:
                samples = list(reader)[:max_samples]  # Limit samples
            task_data[config_name] = samples
            logger.info(f"  {config_name}: {len(samples)} samples")
        else:
            logger.warning(f"  {config_name}: File not found")
    
    return {task: task_data}


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


def match_samples_simple(task_data):
    """Simple sample matching by index."""
    if 'base' not in task_data:
        return []
    
    min_samples = min(len(samples) for samples in task_data.values())
    logger.info(f"Matching {min_samples} samples by index...")
    
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
        
        response = sample['resps'][0][0] if sample['resps'] else ""
        
        # Simple text-based features
        features = {
            'problem_length': len(problem.split()),
            'problem_chars': len(problem),
            'response_length': len(response.split()),
            'response_chars': len(response),
            'math_symbols': problem.count('$') + problem.count('\\'),
            'has_frac': 1 if 'frac' in problem else 0,
            'has_sqrt': 1 if 'sqrt' in problem else 0,
            'problem_lines': len(problem.split('\n')),
            'response_lines': len(response.split('\n')),
            'response_ratio': len(response) / max(1, len(problem))
        }
        
        return features
        
    except Exception as e:
        logger.debug(f"Feature extraction error: {e}")
        return None


def train_simple_classifiers(task_data):
    """Train simple classifiers using text features."""
    logger.info("=== Training Simple Classifiers ===")
    
    matched_samples = match_samples_simple(task_data)
    logger.info(f"Training on {len(matched_samples)} matched samples")
    
    config_tokens = {
        'enhanced100': 100,
        'enhanced500': 500,
        'enhanced1000': 1000
    }
    
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
                
                # Extract simple features
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
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
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
            'positive_rate': positive_rate
        }
    
    return task_results


def evaluate_progressive_strategy(task_data, trained_models, threshold=0.5):
    """Evaluate the progressive strategy."""
    logger.info("=== Evaluating Progressive Strategy ===")
    
    matched_samples = match_samples_simple(task_data)
    
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
                if config_name in trained_models:
                    # Extract features
                    features = extract_simple_features(sample[config_name])
                    
                    if features is not None:
                        # Make prediction
                        model_data = trained_models[config_name]
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
        logger.info(f"Progressive accuracy: {accuracy:.3f} ({total_correct}/{total_samples})")
        logger.info(f"Strategy usage: {strategy_usage}")
        
        # Calculate average tokens used for assistance
        total_usage = sum(strategy_usage.values())
        if total_usage > 0:
            avg_tokens = (strategy_usage['enhanced100'] * 100 + 
                         strategy_usage['enhanced500'] * 500 + 
                         strategy_usage['enhanced1000'] * 1000) / total_usage
            logger.info(f"Average assistance tokens: {avg_tokens:.0f}")
            
            assistance_rate = (total_usage - strategy_usage['base']) / total_usage
            logger.info(f"Assistance rate: {assistance_rate:.1%}")
        
        return accuracy, strategy_usage
    
    return 0.0, {}


def calculate_baselines(task_data):
    """Calculate baseline accuracies."""
    logger.info("=== Calculating Baselines ===")
    
    matched_samples = match_samples_simple(task_data)
    
    baselines = {
        'base_only': 0,
        'always_100': 0,
        'always_500': 0,
        'always_1000': 0,
        'optimal': 0
    }
    
    for sample in matched_samples:
        # Evaluate correctness for all strategies
        results = {}
        for cfg in ['base', 'enhanced100', 'enhanced500', 'enhanced1000']:
            if cfg in sample:
                results[cfg] = evaluate_correctness(sample[cfg])
        
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
    
    logger.info("Baseline accuracies:")
    for strategy, accuracy in baselines.items():
        logger.info(f"  {strategy}: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return baselines


def main():
    parser = argparse.ArgumentParser(description='Fast Progressive Assistance Predictor Test')
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser.add_argument('--results-dir', default=os.path.join(project_root, 'results'),
                       help='Results directory containing actual model outputs')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Maximum samples to use for fast testing')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for predictions')
    
    args = parser.parse_args()
    
    # Load sample data
    logger.info("Loading sample data for fast testing...")
    all_data = load_sample_data(args.results_dir, args.max_samples)
    
    if not all_data:
        logger.error("No data loaded!")
        return
    
    task = list(all_data.keys())[0]
    task_data = all_data[task]
    
    # Calculate baselines
    baselines = calculate_baselines(task_data)
    
    # Train simple classifiers
    trained_models = train_simple_classifiers(task_data)
    
    if not trained_models:
        logger.error("No classifiers trained!")
        return
    
    # Evaluate progressive strategy
    accuracy, strategy_usage = evaluate_progressive_strategy(
        task_data, trained_models, args.threshold
    )
    
    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"{'FAST PROGRESSIVE ASSISTANCE PREDICTOR':^60}")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Task: {task.upper()}")
    logger.info(f"📈 Samples: {args.max_samples}")
    logger.info(f"🎯 Logic: Try 100 → 500 → 1000 → Base only")
    
    logger.info(f"\n📊 BASELINE ACCURACIES:")
    logger.info(f"   7B Only:        {baselines['base_only']:.3f} ({baselines['base_only']*100:.1f}%)")
    logger.info(f"   Always 100:     {baselines['always_100']:.3f} ({baselines['always_100']*100:.1f}%)")
    logger.info(f"   Always 500:     {baselines['always_500']:.3f} ({baselines['always_500']*100:.1f}%)")
    logger.info(f"   Always 1000:    {baselines['always_1000']:.3f} ({baselines['always_1000']*100:.1f}%)")
    logger.info(f"   Optimal:        {baselines['optimal']:.3f} ({baselines['optimal']*100:.1f}%)")
    
    logger.info(f"\n🧠 PROGRESSIVE PREDICTOR:")
    logger.info(f"   Accuracy:       {accuracy:.3f} ({accuracy*100:.1f}%)")
    logger.info(f"   Strategy usage: {strategy_usage}")
    
    if sum(strategy_usage.values()) > 0:
        avg_tokens = (strategy_usage['enhanced100'] * 100 + 
                     strategy_usage['enhanced500'] * 500 + 
                     strategy_usage['enhanced1000'] * 1000) / sum(strategy_usage.values())
        logger.info(f"   Avg tokens:     {avg_tokens:.0f}")
    
    # Performance analysis
    improvement = accuracy - baselines['base_only']
    logger.info(f"\n📈 ANALYSIS:")
    logger.info(f"   Improvement:    +{improvement:.3f} ({improvement*100:.1f}pp)")
    
    best_fixed = max(baselines['always_100'], baselines['always_500'], baselines['always_1000'])
    if accuracy > best_fixed:
        logger.info(f"   ✅ Progressive > Best fixed ({accuracy:.3f} vs {best_fixed:.3f})")
    else:
        logger.info(f"   ⚠️  Progressive ≤ Best fixed ({accuracy:.3f} vs {best_fixed:.3f})")
    
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()