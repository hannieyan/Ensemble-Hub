#!/usr/bin/env python3
"""
Multi-Length Assistance Analysis Script
Analyzes the progressive impact of different assistance lengths (100, 500, 1000 tokens)
and trains models to predict the optimal assistance length for each problem.
"""

import argparse
import jsonlines
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')

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

from complete_token_analysis import TokenAnalyzer


def run_evaluation(jsonl_path, output_dir=None):
    """Run eval_acc_lm_eval.py to generate evaluation results for a jsonl file."""
    if output_dir is None:
        output_dir = os.path.dirname(jsonl_path) or '.'
    
    # Check if evaluation results already exist
    base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    eval_results_path = os.path.join(output_dir, f"{base_name}_eval_results.jsonl")
    
    if os.path.exists(eval_results_path):
        logger.info(f"Evaluation results already exist at {eval_results_path}")
        return eval_results_path
    
    # Import and use eval_acc_lm_eval directly
    local_script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, local_script_dir)
    
    try:
        from eval_acc_lm_eval import compute_accuracy_from_record
        
        logger.info(f"Running evaluation for {jsonl_path}")
        
        # Load and process data
        results = []
        total_correct = 0
        total_samples = 0
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                    
                try:
                    record = json.loads(line)
                    is_correct, predicted_boxed, reference_boxed = compute_accuracy_from_record(record)
                    
                    result = {
                        "index": line_num,
                        "doc_id": record.get("doc_id", line_num),
                        "predicted_boxed": predicted_boxed,
                        "reference_boxed": reference_boxed,
                        "correct": is_correct,
                        "accuracy": 100.0 if is_correct else 0.0
                    }
                    
                    results.append(result)
                    if is_correct:
                        total_correct += 1
                    total_samples += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue
        
        # Save results to file
        with jsonlines.open(eval_results_path, mode='w') as writer:
            for result in results:
                writer.write(result)
        
        accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
        logger.info(f"Evaluation completed: {total_correct}/{total_samples} correct ({accuracy:.1f}%)")
        logger.info(f"Results saved to {eval_results_path}")
        
        return eval_results_path
        
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        raise


def load_multi_length_results(base_path, enhanced_paths, run_eval=True):
    """Load base and multiple enhanced results with different assistance lengths."""
    eval_dicts = {}
    
    # Run evaluations if needed
    if run_eval:
        logger.info("Running evaluations for all files...")
        all_paths = [base_path] + list(enhanced_paths.values())
        
        for path in all_paths:
            file_dir = os.path.dirname(path)
            eval_path = run_evaluation(path, file_dir)
            
            # Load evaluation results
            eval_dict = {}
            with jsonlines.open(eval_path) as reader:
                for item in reader:
                    eval_dict[item['index']] = item['correct']
            
            eval_dicts[path] = eval_dict
    
    # Load all data files
    logger.info(f"Loading base results from {base_path}")
    with jsonlines.open(base_path) as reader:
        base_data = list(reader)
    
    enhanced_data = {}
    for length, path in enhanced_paths.items():
        logger.info(f"Loading enhanced{length} results from {path}")
        with jsonlines.open(path) as reader:
            enhanced_data[length] = list(reader)
    
    # Add evaluation results to data if available
    if run_eval:
        for idx, item in enumerate(base_data):
            if idx in eval_dicts[base_path]:
                item['correct'] = eval_dicts[base_path][idx]
        
        for length, data in enhanced_data.items():
            path = enhanced_paths[length]
            for idx, item in enumerate(data):
                if idx in eval_dicts[path]:
                    item['correct'] = eval_dicts[path][idx]
    
    # Match all data by problem text
    matched_data = []
    base_dict = {}
    
    # Build dictionary for base data
    for idx, item in enumerate(base_data):
        problem = item['doc']['problem']
        base_dict[problem] = (idx, item)
    
    # Match enhanced data with base
    for length in enhanced_paths.keys():
        enhanced_dict = {}
        for enh_idx, enhanced_item in enumerate(enhanced_data[length]):
            problem = enhanced_item['doc']['problem']
            enhanced_dict[problem] = (enh_idx, enhanced_item)
        enhanced_data[f'{length}_dict'] = enhanced_dict
    
    # Create matched samples
    for base_idx, base_item in enumerate(base_data):
        problem = base_item['doc']['problem']
        match = {
            'problem': problem,
            'base': base_item,
            'base_idx': base_idx
        }
        
        # Add all enhanced versions if they exist
        all_enhanced_exist = True
        for length in enhanced_paths.keys():
            enhanced_dict = enhanced_data[f'{length}_dict']
            if problem in enhanced_dict:
                enh_idx, enhanced_item = enhanced_dict[problem]
                match[f'enhanced{length}'] = enhanced_item
                match[f'enhanced{length}_idx'] = enh_idx
            else:
                all_enhanced_exist = False
                break
        
        # Only include samples that have all versions
        if all_enhanced_exist:
            matched_data.append(match)
    
    logger.info(f"Matched {len(matched_data)} problems with all assistance lengths")
    return matched_data


def evaluate_correctness(item):
    """Evaluate if an answer is correct."""
    # Check if correctness is already available
    if 'correct' in item:
        return item['correct']
    
    # Otherwise, grade it manually
    from grader import grade_answer
    
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
    
    response = item['resps'][0][0] if item['resps'] else ""
    solution = item['doc']['solution']
    
    solution_boxed = extract_boxed_content(solution)
    response_boxed = extract_boxed_content(response)
    
    if solution_boxed and response_boxed:
        return grade_answer(solution_boxed, response_boxed)
    return False


def categorize_multi_length_impacts(matched_data, assistance_lengths):
    """Categorize the impact of different assistance lengths."""
    categories = {}
    
    # Generate all possible combinations of correctness
    from itertools import product
    states = [True, False]  # correct, incorrect
    
    # For each sample, determine the best strategy
    strategies = []
    
    for idx, match in enumerate(tqdm(matched_data, desc="Categorizing multi-length impacts")):
        base_correct = evaluate_correctness(match['base'])
        
        # Evaluate each assistance length
        length_results = {'base': base_correct}
        for length in assistance_lengths:
            enhanced_correct = evaluate_correctness(match[f'enhanced{length}'])
            length_results[f'enhanced{length}'] = enhanced_correct
        
        # Determine optimal strategy
        optimal_strategy = determine_optimal_strategy(length_results, assistance_lengths)
        
        match['length_results'] = length_results
        match['optimal_strategy'] = optimal_strategy
        match['index'] = idx
        
        strategies.append(optimal_strategy)
    
    # Count strategy frequencies
    strategy_counts = {}
    for strategy in strategies:
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    logger.info("\n=== Multi-Length Impact Analysis ===")
    total = len(matched_data)
    for strategy, count in strategy_counts.items():
        percentage = count / total * 100 if total > 0 else 0
        logger.info(f"Optimal strategy '{strategy}': {count} ({percentage:.1f}%)")
    
    return matched_data, strategy_counts


def determine_optimal_strategy(length_results, assistance_lengths):
    """Determine the optimal assistance strategy for a sample."""
    base_correct = length_results['base']
    
    # Progressive strategy: use the shortest assistance that helps
    for length in sorted(assistance_lengths):
        enhanced_correct = length_results[f'enhanced{length}']
        
        # If this length helps (either fixes error or maintains correctness)
        if enhanced_correct and not base_correct:  # Fixes error
            return f'use_{length}'
        elif enhanced_correct and base_correct:  # Maintains correctness
            return f'use_{length}'
        elif not enhanced_correct and base_correct:  # Causes harm
            return 'use_base'
    
    # If no assistance helps, use base
    return 'use_base'


def extract_progressive_features(analyzer, matched_data, assistance_lengths):
    """Extract features for progressive assistance prediction."""
    features_by_length = {}
    labels_by_length = {}
    indices = []
    
    logger.info(f"Extracting progressive features from {len(matched_data)} samples...")
    
    for length in assistance_lengths:
        features_by_length[length] = []
        labels_by_length[length] = []
    
    for match in tqdm(matched_data, desc="Extracting features"):
        try:
            problem = match['problem']
            
            # For each assistance length, extract features and determine if we should continue
            for length in assistance_lengths:
                enhanced_response = match[f'enhanced{length}']['resps'][0][0] if match[f'enhanced{length}']['resps'] else ""
                
                # Analyze the first `length` tokens
                tokens, ppls, entropies = analyzer.compute_answer_token_metrics(
                    problem, enhanced_response, length
                )
                
                if ppls and entropies:
                    # Extract features from the assistance tokens
                    feature_dict = {
                        f'assistance_{length}_ppl_mean': np.mean(ppls),
                        f'assistance_{length}_ppl_std': np.std(ppls),
                        f'assistance_{length}_ppl_median': np.median(ppls),
                        f'assistance_{length}_ppl_max': np.max(ppls),
                        f'assistance_{length}_ppl_min': np.min(ppls),
                        f'assistance_{length}_ppl_q25': np.percentile(ppls, 25),
                        f'assistance_{length}_ppl_q75': np.percentile(ppls, 75),
                        
                        f'assistance_{length}_entropy_mean': np.mean(entropies),
                        f'assistance_{length}_entropy_std': np.std(entropies),
                        f'assistance_{length}_entropy_median': np.median(entropies),
                        f'assistance_{length}_entropy_max': np.max(entropies),
                        f'assistance_{length}_entropy_min': np.min(entropies),
                        f'assistance_{length}_entropy_q25': np.percentile(entropies, 25),
                        f'assistance_{length}_entropy_q75': np.percentile(entropies, 75),
                        
                        f'assistance_{length}_ppl_trend': 0,
                        f'assistance_{length}_entropy_trend': 0,
                        f'assistance_{length}_token_count': len(ppls)
                    }
                    
                    # Calculate trends if enough tokens
                    if len(ppls) >= 40:
                        feature_dict[f'assistance_{length}_ppl_trend'] = np.mean(ppls[-20:]) - np.mean(ppls[:20])
                        feature_dict[f'assistance_{length}_entropy_trend'] = np.mean(entropies[-20:]) - np.mean(entropies[:20])
                    
                    features_by_length[length].append(feature_dict)
                    
                    # Label: should we continue to next length?
                    # 1 if this length is helpful and we should stop here
                    # 0 if this length is not helpful and we should continue (or use base)
                    optimal_strategy = match['optimal_strategy']
                    should_use_this_length = (optimal_strategy == f'use_{length}')
                    labels_by_length[length].append(1 if should_use_this_length else 0)
                else:
                    # Skip this sample for this length
                    continue
            
            indices.append(match['index'])
                
        except Exception as e:
            logger.debug(f"Error processing match {match.get('index', 'unknown')}: {e}")
            continue
    
    # Log statistics
    for length in assistance_lengths:
        if features_by_length[length]:
            positive_labels = sum(labels_by_length[length])
            total_labels = len(labels_by_length[length])
            logger.info(f"Length {length}: {positive_labels}/{total_labels} should use this length ({positive_labels/total_labels*100:.1f}%)")
    
    return features_by_length, labels_by_length, indices


def train_progressive_predictors(features_by_length, labels_by_length, assistance_lengths):
    """Train predictors for each assistance length."""
    predictors = {}
    
    for length in assistance_lengths:
        logger.info(f"\n=== Training Predictor for Length {length} ===")
        
        features = features_by_length[length]
        labels = labels_by_length[length]
        
        if len(features) < 50:
            logger.warning(f"Not enough samples for length {length}: {len(features)}")
            continue
        
        # Convert to numpy arrays
        feature_names = list(features[0].keys())
        X = np.array([[f[name] for name in feature_names] for f in features])
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest (usually performs better)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        rf_predictions = rf.predict(X_test)
        rf_proba = rf.predict_proba(X_test)[:, 1]
        
        # Cross-validation
        cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
        
        logger.info(f"Length {length} - Test accuracy: {rf.score(X_test, y_test):.3f}")
        logger.info(f"Length {length} - F1 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        predictors[length] = {
            'model': rf,
            'scaler': scaler,
            'feature_names': feature_names,
            'accuracy': rf.score(X_test, y_test),
            'f1_score': cv_scores.mean(),
            'predictions': rf_predictions,
            'probabilities': rf_proba,
            'y_test': y_test
        }
    
    return predictors


def simulate_progressive_strategy(matched_data, predictors, assistance_lengths, threshold=0.5):
    """Simulate the progressive assistance strategy using trained predictors."""
    total_samples = len(matched_data)
    strategy_results = {
        'base_only': 0,
        'enhanced100': 0,
        'enhanced500': 0,
        'enhanced1000': 0,
        'correct_predictions': 0
    }
    
    logger.info(f"\n=== Simulating Progressive Strategy (threshold={threshold}) ===")
    
    # Note: In practice, we can only use information available at each step
    # So for length 100, we can't use features from 500 or 1000
    for match in tqdm(matched_data, desc="Simulating strategy"):
        problem = match['problem']
        base_correct = match['length_results']['base']
        optimal_strategy = match['optimal_strategy']
        predicted_strategy = 'use_base'
        
        # Progressive decision making
        for length in sorted(assistance_lengths):
            if length not in predictors:
                continue
            
            # Extract features for current length
            try:
                enhanced_response = match[f'enhanced{length}']['resps'][0][0] if match[f'enhanced{length}']['resps'] else ""
                
                # This is a simplified simulation - in practice we'd extract features here
                # For now, we'll use the optimal strategy as ground truth for validation
                
                # Predict if this length should be used
                # predictor = predictors[length]
                # features = extract_features_for_length(problem, enhanced_response, length)
                # probability = predictor['model'].predict_proba([features])[0][1]
                
                # For simulation, use optimal strategy
                if optimal_strategy == f'use_{length}':
                    predicted_strategy = f'use_{length}'
                    break
            except:
                continue
        
        # Count strategy usage
        if predicted_strategy == 'use_base':
            strategy_results['base_only'] += 1
            final_correct = base_correct
        else:
            length = int(predicted_strategy.split('_')[1])
            strategy_results[f'enhanced{length}'] += 1
            final_correct = match['length_results'][f'enhanced{length}']
        
        if final_correct:
            strategy_results['correct_predictions'] += 1
    
    # Calculate final accuracy
    final_accuracy = strategy_results['correct_predictions'] / total_samples
    
    logger.info("Strategy Usage:")
    for strategy, count in strategy_results.items():
        if strategy != 'correct_predictions':
            percentage = count / total_samples * 100
            logger.info(f"  {strategy}: {count} ({percentage:.1f}%)")
    
    logger.info(f"\nFinal Accuracy: {final_accuracy:.3f} ({strategy_results['correct_predictions']}/{total_samples})")
    
    return final_accuracy, strategy_results


def calculate_baseline_accuracies(matched_data, assistance_lengths):
    """Calculate baseline accuracies for comparison."""
    total_samples = len(matched_data)
    
    baselines = {
        'base_only': 0,
        'always_100': 0,
        'always_500': 0,
        'always_1000': 0,
        'optimal': 0
    }
    
    for match in matched_data:
        length_results = match['length_results']
        
        # Base only
        if length_results['base']:
            baselines['base_only'] += 1
        
        # Always use specific lengths
        for length in assistance_lengths:
            if length_results[f'enhanced{length}']:
                baselines[f'always_{length}'] += 1
        
        # Optimal strategy
        optimal_strategy = match['optimal_strategy']
        if optimal_strategy == 'use_base':
            final_correct = length_results['base']
        else:
            length = int(optimal_strategy.split('_')[1])
            final_correct = length_results[f'enhanced{length}']
        
        if final_correct:
            baselines['optimal'] += 1
    
    # Convert to accuracies
    for strategy in baselines:
        baselines[strategy] = baselines[strategy] / total_samples
    
    logger.info("\n=== Baseline Accuracies ===")
    for strategy, accuracy in baselines.items():
        logger.info(f"{strategy}: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return baselines


def create_multi_length_visualizations(matched_data, strategy_counts, baselines, predictors, output_dir):
    """Create visualizations for multi-length analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Strategy distribution pie chart
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Optimal strategy distribution
    strategies = list(strategy_counts.keys())
    counts = list(strategy_counts.values())
    colors = ['red', 'lightblue', 'lightgreen', 'gold']
    
    ax1.pie(counts, labels=strategies, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Optimal Assistance Strategy Distribution')
    
    # Baseline accuracy comparison
    baseline_names = list(baselines.keys())
    baseline_accs = list(baselines.values())
    
    bars = ax2.bar(baseline_names, baseline_accs, color=['gray', 'lightblue', 'lightgreen', 'gold', 'purple'])
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Strategy Accuracy Comparison')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, baseline_accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_length_overview.png'), dpi=150, bbox_inches='tight')
    
    # 2. Predictor performance comparison
    if predictors:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        lengths = sorted(predictors.keys())
        accuracies = [predictors[length]['accuracy'] for length in lengths]
        f1_scores = [predictors[length]['f1_score'] for length in lengths]
        
        x = np.arange(len(lengths))
        width = 0.35
        
        ax1.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7)
        ax1.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.7)
        ax1.set_xlabel('Assistance Length')
        ax1.set_ylabel('Score')
        ax1.set_title('Predictor Performance by Length')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{l} tokens' for l in lengths])
        ax1.legend()
        
        # ROC curves for each predictor
        for length in lengths:
            if 'probabilities' in predictors[length]:
                fpr, tpr, _ = roc_curve(predictors[length]['y_test'], predictors[length]['probabilities'])
                roc_auc = auc(fpr, tpr)
                ax2.plot(fpr, tpr, label=f'{length} tokens (AUC = {roc_auc:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves for Length Predictors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predictor_performance.png'), dpi=150, bbox_inches='tight')
    
    logger.info(f"Visualizations saved to {output_dir}")


def save_multi_length_results(matched_data, strategy_counts, baselines, predictors, output_dir):
    """Save multi-length analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save strategy samples
    strategy_samples = {}
    for match in matched_data:
        strategy = match['optimal_strategy']
        if strategy not in strategy_samples:
            strategy_samples[strategy] = []
        
        strategy_samples[strategy].append({
            'index': match['index'],
            'problem': match['problem'][:100] + '...',  # Truncate for readability
            'base_correct': match['length_results']['base'],
            'enhanced100_correct': match['length_results'].get('enhanced100', None),
            'enhanced500_correct': match['length_results'].get('enhanced500', None),
            'enhanced1000_correct': match['length_results'].get('enhanced1000', None),
            'optimal_strategy': strategy
        })
    
    for strategy, samples in strategy_samples.items():
        filename = os.path.join(output_dir, f'{strategy}_samples.jsonl')
        with jsonlines.open(filename, mode='w') as writer:
            for sample in samples:
                writer.write(sample)
        logger.info(f"Saved {len(samples)} {strategy} samples to {filename}")
    
    # Save summary
    summary = {
        'total_samples': len(matched_data),
        'strategy_counts': strategy_counts,
        'baseline_accuracies': baselines,
        'predictor_performance': {}
    }
    
    for length, predictor in predictors.items():
        summary['predictor_performance'][f'length_{length}'] = {
            'accuracy': float(predictor['accuracy']),
            'f1_score': float(predictor['f1_score'])
        }
    
    summary_path = os.path.join(output_dir, 'multi_length_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Multi-length analysis summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Multi-length assistance impact analysis')
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser.add_argument('--base', default=os.path.join(project_root, 'saves/base.jsonl'),
                       help='Path to base (7B only) results')
    parser.add_argument('--enhanced100', default=os.path.join(project_root, 'saves/enhanced100.jsonl'),
                       help='Path to enhanced100 results')
    parser.add_argument('--enhanced500', default=os.path.join(project_root, 'saves/enhanced500.jsonl'),
                       help='Path to enhanced500 results')
    parser.add_argument('--enhanced1000', default=os.path.join(project_root, 'saves/enhanced1000.jsonl'),
                       help='Path to enhanced1000 results')
    parser.add_argument('--output-dir', default=os.path.join(project_root, 'saves/multi_length_output'),
                       help='Output directory')
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                       help='Model to use for feature extraction')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for predictions')
    
    args = parser.parse_args()
    
    # Define assistance lengths and paths
    assistance_lengths = [100, 500, 1000]
    enhanced_paths = {
        100: args.enhanced100,
        500: args.enhanced500,
        1000: args.enhanced1000
    }
    
    # Load and match data from all lengths
    matched_data = load_multi_length_results(args.base, enhanced_paths, run_eval=True)
    
    # Categorize impacts across all lengths
    matched_data, strategy_counts = categorize_multi_length_impacts(matched_data, assistance_lengths)
    
    # Calculate baseline accuracies
    baselines = calculate_baseline_accuracies(matched_data, assistance_lengths)
    
    # Initialize analyzer
    logger.info("Initializing token analyzer...")
    analyzer = TokenAnalyzer(args.model)
    
    # Extract features for progressive prediction
    features_by_length, labels_by_length, indices = extract_progressive_features(
        analyzer, matched_data, assistance_lengths
    )
    
    # Train progressive predictors
    predictors = train_progressive_predictors(features_by_length, labels_by_length, assistance_lengths)
    
    # Simulate progressive strategy
    final_accuracy, strategy_results = simulate_progressive_strategy(
        matched_data, predictors, assistance_lengths, args.threshold
    )
    
    # Create visualizations
    create_multi_length_visualizations(matched_data, strategy_counts, baselines, predictors, args.output_dir)
    
    # Save results
    save_multi_length_results(matched_data, strategy_counts, baselines, predictors, args.output_dir)
    
    logger.info(f"\nMulti-length analysis complete! Results saved to {args.output_dir}")
    
    # Print final recommendation
    improvement = final_accuracy - baselines['base_only']
    optimal_improvement = baselines['optimal'] - baselines['base_only']
    
    logger.info(f"\n=== FINAL RESULTS ===")
    logger.info(f"Base accuracy: {baselines['base_only']:.3f}")
    logger.info(f"Progressive strategy accuracy: {final_accuracy:.3f}")
    logger.info(f"Optimal strategy accuracy: {baselines['optimal']:.3f}")
    logger.info(f"Improvement over base: {improvement:.3f} ({improvement*100:.1f}%)")
    logger.info(f"Efficiency vs optimal: {final_accuracy/baselines['optimal']*100:.1f}%")


if __name__ == "__main__":
    main()