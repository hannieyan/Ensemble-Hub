#!/usr/bin/env python3
"""
Analyze the impact of 32B model assistance on 7B model performance.
Identifies cases where assistance helps or hurts, and trains a predictor
to filter out harmful assistance.
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    
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


def load_and_match_results(base_path, enhanced_path, run_eval=True):
    """Load base and enhanced results and match them by problem."""
    base_eval_dict = {}
    enhanced_eval_dict = {}
    
    # Run evaluations if needed
    if run_eval:
        logger.info("Running evaluations if needed...")
        base_dir = os.path.dirname(base_path)
        enhanced_dir = os.path.dirname(enhanced_path)
        
        # Run evaluation for base
        base_eval_path = run_evaluation(base_path, base_dir)
        
        # Run evaluation for enhanced
        enhanced_eval_path = run_evaluation(enhanced_path, enhanced_dir)
        
        # Load evaluation results
        with jsonlines.open(base_eval_path) as reader:
            for item in reader:
                base_eval_dict[item['index']] = item['correct']
        
        with jsonlines.open(enhanced_eval_path) as reader:
            for item in reader:
                enhanced_eval_dict[item['index']] = item['correct']
    
    logger.info(f"Loading base results from {base_path}")
    with jsonlines.open(base_path) as reader:
        base_data = list(reader)
    
    logger.info(f"Loading enhanced results from {enhanced_path}")
    with jsonlines.open(enhanced_path) as reader:
        enhanced_data = list(reader)
    
    # Add evaluation results to data if available
    if run_eval:
        for idx, item in enumerate(base_data):
            if idx in base_eval_dict:
                item['correct'] = base_eval_dict[idx]
        
        for idx, item in enumerate(enhanced_data):
            if idx in enhanced_eval_dict:
                item['correct'] = enhanced_eval_dict[idx]
    
    # Match by problem text
    matched_data = []
    base_dict = {}
    
    # Build dictionary for base data
    for idx, item in enumerate(base_data):
        problem = item['doc']['problem']
        base_dict[problem] = (idx, item)
    
    # Match enhanced data with base
    for enh_idx, enhanced_item in enumerate(enhanced_data):
        problem = enhanced_item['doc']['problem']
        if problem in base_dict:
            base_idx, base_item = base_dict[problem]
            matched_data.append({
                'problem': problem,
                'base': base_item,
                'enhanced': enhanced_item,
                'base_idx': base_idx,
                'enhanced_idx': enh_idx
            })
    
    logger.info(f"Matched {len(matched_data)} problems")
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


def categorize_impacts(matched_data):
    """Categorize the impact of assistance on each problem."""
    categories = {
        'correct_to_correct': [],  # Good: maintained correctness
        'correct_to_incorrect': [],  # BAD: assistance hurt
        'incorrect_to_correct': [],  # GREAT: assistance helped
        'incorrect_to_incorrect': []  # Neutral: still wrong
    }
    
    for idx, match in enumerate(tqdm(matched_data, desc="Categorizing impacts")):
        base_correct = evaluate_correctness(match['base'])
        enhanced_correct = evaluate_correctness(match['enhanced'])
        
        match['base_correct'] = base_correct
        match['enhanced_correct'] = enhanced_correct
        match['index'] = idx
        
        if base_correct and enhanced_correct:
            categories['correct_to_correct'].append(match)
        elif base_correct and not enhanced_correct:
            categories['correct_to_incorrect'].append(match)
        elif not base_correct and enhanced_correct:
            categories['incorrect_to_correct'].append(match)
        else:
            categories['incorrect_to_incorrect'].append(match)
    
    # Print statistics
    logger.info("\n=== Impact Analysis ===")
    total = len(matched_data)
    for category, items in categories.items():
        count = len(items)
        percentage = count / total * 100 if total > 0 else 0
        logger.info(f"{category}: {count} ({percentage:.1f}%)")
    
    return categories


def extract_features_for_prediction(analyzer, matched_data, n_tokens=1000):
    """Extract features from the 32B assistance tokens to predict impact."""
    features = []
    labels = []  # 1 if assistance will hurt (correct->incorrect), 0 otherwise
    indices = []
    
    logger.info(f"Extracting features from {len(matched_data)} samples...")
    
    for match in tqdm(matched_data, desc="Extracting features"):
        try:
            # Get the enhanced response (which includes 32B assistance)
            enhanced_response = match['enhanced']['resps'][0][0] if match['enhanced']['resps'] else ""
            problem = match['problem']
            
            # Analyze the first n_tokens (the 32B assistance part)
            tokens, ppls, entropies = analyzer.compute_answer_token_metrics(
                problem, enhanced_response, n_tokens
            )
            
            if ppls and entropies:
                # Extract features from the assistance tokens
                feature_dict = {
                    # PPL features
                    'assistance_ppl_mean': np.mean(ppls),
                    'assistance_ppl_std': np.std(ppls),
                    'assistance_ppl_median': np.median(ppls),
                    'assistance_ppl_max': np.max(ppls),
                    'assistance_ppl_min': np.min(ppls),
                    'assistance_ppl_q25': np.percentile(ppls, 25),
                    'assistance_ppl_q75': np.percentile(ppls, 75),
                    
                    # Entropy features
                    'assistance_entropy_mean': np.mean(entropies),
                    'assistance_entropy_std': np.std(entropies),
                    'assistance_entropy_median': np.median(entropies),
                    'assistance_entropy_max': np.max(entropies),
                    'assistance_entropy_min': np.min(entropies),
                    'assistance_entropy_q25': np.percentile(entropies, 25),
                    'assistance_entropy_q75': np.percentile(entropies, 75),
                    
                    # Trend features
                    'assistance_ppl_trend': 0,
                    'assistance_entropy_trend': 0,
                    
                    # Token count
                    'assistance_token_count': len(ppls)
                }
                
                # Calculate trends if enough tokens
                if len(ppls) >= 40:
                    feature_dict['assistance_ppl_trend'] = np.mean(ppls[-20:]) - np.mean(ppls[:20])
                    feature_dict['assistance_entropy_trend'] = np.mean(entropies[-20:]) - np.mean(entropies[:20])
                
                features.append(feature_dict)
                
                # Label is 1 if assistance hurt (correct->incorrect)
                is_harmful = match['base_correct'] and not match['enhanced_correct']
                labels.append(1 if is_harmful else 0)
                indices.append(match['index'])
                
        except Exception as e:
            logger.debug(f"Error processing match {match.get('index', 'unknown')}: {e}")
            continue
    
    logger.info(f"Extracted features from {len(features)} samples")
    logger.info(f"Harmful assistance cases: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    
    return features, labels, indices


def train_harm_predictor(features, labels):
    """Train models to predict when assistance will be harmful."""
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
    
    results = {}
    
    # 1. Logistic Regression
    logger.info("\n=== Training Harm Predictor ===")
    logger.info("\n1. Logistic Regression")
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    lr_predictions = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    
    results['logistic'] = {
        'model': lr,
        'accuracy': lr.score(X_test_scaled, y_test),
        'predictions': lr_predictions,
        'probabilities': lr_proba,
        'feature_importance': dict(zip(feature_names, lr.coef_[0]))
    }
    
    # 2. Random Forest
    logger.info("\n2. Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    
    results['random_forest'] = {
        'model': rf,
        'accuracy': rf.score(X_test, y_test),
        'predictions': rf_predictions,
        'probabilities': rf_proba,
        'feature_importance': dict(zip(feature_names, rf.feature_importances_))
    }
    
    # Print classification reports
    for name, result in results.items():
        logger.info(f"\n{name} Classification Report:")
        logger.info(classification_report(y_test, result['predictions'], 
                                        target_names=['Not Harmful', 'Harmful']))
    
    # Cross-validation
    logger.info("\n3. Cross-validation scores (5-fold)")
    cv_scores_lr = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='f1')
    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
    
    logger.info(f"Logistic Regression F1: {cv_scores_lr.mean():.3f} (+/- {cv_scores_lr.std() * 2:.3f})")
    logger.info(f"Random Forest F1: {cv_scores_rf.mean():.3f} (+/- {cv_scores_rf.std() * 2:.3f})")
    
    results['X_test'] = X_test
    results['y_test'] = y_test
    results['scaler'] = scaler
    results['feature_names'] = feature_names
    
    return results


def calculate_filtered_accuracy(categories, predictor_results, features, indices, threshold=0.5):
    """Calculate accuracy after filtering out predicted harmful cases."""
    # Get the best predictor (random forest usually performs better)
    predictor = predictor_results['random_forest']['model']
    scaler = predictor_results['scaler']
    feature_names = predictor_results['feature_names']
    
    # Prepare all features
    all_features = np.array([[f[name] for name in feature_names] for f in features])
    
    # Get predictions for all samples
    if 'logistic' in str(type(predictor)):
        all_features_scaled = scaler.transform(all_features)
        harm_probabilities = predictor.predict_proba(all_features_scaled)[:, 1]
    else:
        harm_probabilities = predictor.predict_proba(all_features)[:, 1]
    
    # Create index to probability mapping
    index_to_prob = {idx: prob for idx, prob in zip(indices, harm_probabilities)}
    
    # Calculate different accuracy scenarios
    total_samples = sum(len(items) for items in categories.values())
    
    # Base accuracy (7B only)
    base_correct = len(categories['correct_to_correct']) + len(categories['correct_to_incorrect'])
    base_accuracy = base_correct / total_samples
    
    # Enhanced accuracy (with 32B assistance)
    enhanced_correct = len(categories['correct_to_correct']) + len(categories['incorrect_to_correct'])
    enhanced_accuracy = enhanced_correct / total_samples
    
    # Filtered accuracy (use assistance only when predicted not harmful)
    filtered_correct = 0
    filtered_stats = {
        'kept_assistance': 0,
        'rejected_assistance': 0,
        'avoided_harm': 0,
        'missed_help': 0
    }
    
    for category, items in categories.items():
        for item in items:
            idx = item['index']
            harm_prob = index_to_prob.get(idx, 0)
            
            if harm_prob < threshold:  # Keep assistance
                filtered_stats['kept_assistance'] += 1
                if item['enhanced_correct']:
                    filtered_correct += 1
            else:  # Reject assistance, use base result
                filtered_stats['rejected_assistance'] += 1
                if item['base_correct']:
                    filtered_correct += 1
                
                # Track avoided harm and missed help
                if category == 'correct_to_incorrect':
                    filtered_stats['avoided_harm'] += 1
                elif category == 'incorrect_to_correct':
                    filtered_stats['missed_help'] += 1
    
    filtered_accuracy = filtered_correct / total_samples
    
    logger.info("\n=== Accuracy Analysis ===")
    logger.info(f"Base (7B only) accuracy: {base_accuracy:.3f} ({base_correct}/{total_samples})")
    logger.info(f"Enhanced (always use 32B) accuracy: {enhanced_accuracy:.3f} ({enhanced_correct}/{total_samples})")
    logger.info(f"Filtered (selective 32B) accuracy: {filtered_accuracy:.3f} ({filtered_correct}/{total_samples})")
    logger.info(f"\nFiltering Statistics:")
    logger.info(f"  Kept assistance: {filtered_stats['kept_assistance']} ({filtered_stats['kept_assistance']/total_samples*100:.1f}%)")
    logger.info(f"  Rejected assistance: {filtered_stats['rejected_assistance']} ({filtered_stats['rejected_assistance']/total_samples*100:.1f}%)")
    logger.info(f"  Avoided harmful cases: {filtered_stats['avoided_harm']}")
    logger.info(f"  Missed helpful cases: {filtered_stats['missed_help']}")
    
    return {
        'base_accuracy': base_accuracy,
        'enhanced_accuracy': enhanced_accuracy,
        'filtered_accuracy': filtered_accuracy,
        'filtered_stats': filtered_stats,
        'threshold': threshold
    }


def create_visualizations(categories, predictor_results, features, labels, output_dir):
    """Create visualizations for the analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Impact distribution pie chart
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart of categories
    category_counts = {k: len(v) for k, v in categories.items()}
    colors = ['green', 'red', 'blue', 'gray']
    labels_pie = ['Correct→Correct', 'Correct→Incorrect', 'Incorrect→Correct', 'Incorrect→Incorrect']
    sizes = [category_counts['correct_to_correct'], 
             category_counts['correct_to_incorrect'],
             category_counts['incorrect_to_correct'],
             category_counts['incorrect_to_incorrect']]
    
    ax1.pie(sizes, labels=labels_pie, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Impact of 32B Assistance on 7B Performance')
    
    # Bar chart comparing accuracies
    accuracy_types = ['Base (7B)', 'Enhanced (32B+7B)', 'Filtered (Selective)']
    accuracies = [0.8, 0.85, 0.9]  # Placeholder, will be updated
    bars = ax2.bar(accuracy_types, accuracies, color=['lightblue', 'lightgreen', 'gold'])
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Comparison')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'impact_overview.png'), dpi=150, bbox_inches='tight')
    
    # 2. Feature importance for harm prediction
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Logistic regression coefficients
    lr_importance = predictor_results['logistic']['feature_importance']
    sorted_features = sorted(lr_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    features_lr, coefs = zip(*sorted_features)
    
    y_pos = np.arange(len(features_lr))
    ax1.barh(y_pos, coefs, color=['red' if c > 0 else 'green' for c in coefs])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features_lr)
    ax1.set_xlabel('Coefficient (positive = more harmful)')
    ax1.set_title('Logistic Regression: Harm Predictors')
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Random forest importance
    rf_importance = predictor_results['random_forest']['feature_importance']
    sorted_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    features_rf, importances = zip(*sorted_features)
    
    y_pos = np.arange(len(features_rf))
    ax2.barh(y_pos, importances, color='lightblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features_rf)
    ax2.set_xlabel('Importance')
    ax2.set_title('Random Forest: Harm Predictors')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'harm_predictors.png'), dpi=150, bbox_inches='tight')
    
    # 3. PPL/Entropy distributions for harmful vs non-harmful
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Separate features by harm label
    harmful_features = [f for f, l in zip(features, labels) if l == 1]
    safe_features = [f for f, l in zip(features, labels) if l == 0]
    
    # PPL distribution
    ax1 = axes[0, 0]
    if harmful_features:
        harmful_ppls = [f['assistance_ppl_mean'] for f in harmful_features]
        ax1.hist(harmful_ppls, bins=30, alpha=0.6, label='Harmful', color='red', density=True)
    if safe_features:
        safe_ppls = [f['assistance_ppl_mean'] for f in safe_features]
        ax1.hist(safe_ppls, bins=30, alpha=0.6, label='Safe', color='green', density=True)
    ax1.set_xlabel('Assistance PPL (mean)')
    ax1.set_ylabel('Density')
    ax1.set_title('PPL Distribution by Harm Potential')
    ax1.legend()
    
    # Entropy distribution
    ax2 = axes[0, 1]
    if harmful_features:
        harmful_entropy = [f['assistance_entropy_mean'] for f in harmful_features]
        ax2.hist(harmful_entropy, bins=30, alpha=0.6, label='Harmful', color='red', density=True)
    if safe_features:
        safe_entropy = [f['assistance_entropy_mean'] for f in safe_features]
        ax2.hist(safe_entropy, bins=30, alpha=0.6, label='Safe', color='green', density=True)
    ax2.set_xlabel('Assistance Entropy (mean)')
    ax2.set_ylabel('Density')
    ax2.set_title('Entropy Distribution by Harm Potential')
    ax2.legend()
    
    # ROC curve
    ax3 = axes[1, 0]
    for name, result in [('Logistic Regression', predictor_results['logistic']), 
                        ('Random Forest', predictor_results['random_forest'])]:
        if 'probabilities' in result:
            fpr, tpr, _ = roc_curve(predictor_results['y_test'], result['probabilities'])
            roc_auc = auc(fpr, tpr)
            ax3.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax3.plot([0, 1], [0, 1], 'k--', label='Random')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curves for Harm Detection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Precision-Recall curve
    ax4 = axes[1, 1]
    for name, result in [('Logistic Regression', predictor_results['logistic']), 
                        ('Random Forest', predictor_results['random_forest'])]:
        if 'probabilities' in result:
            precision, recall, _ = precision_recall_curve(predictor_results['y_test'], result['probabilities'])
            ax4.plot(recall, precision, label=name)
    
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision-Recall Curves for Harm Detection')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'harm_analysis.png'), dpi=150, bbox_inches='tight')
    
    logger.info(f"Visualizations saved to {output_dir}")


def save_results(categories, predictor_results, accuracy_results, output_dir):
    """Save analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save category lists
    for category, items in categories.items():
        filename = os.path.join(output_dir, f'{category}_samples.jsonl')
        with jsonlines.open(filename, mode='w') as writer:
            for item in items:
                writer.write({
                    'index': item['index'],
                    'problem': item['problem'],
                    'base_correct': item['base_correct'],
                    'enhanced_correct': item['enhanced_correct']
                })
        logger.info(f"Saved {len(items)} {category} samples to {filename}")
    
    # Save summary
    summary = {
        'total_samples': sum(len(items) for items in categories.values()),
        'category_counts': {k: len(v) for k, v in categories.items()},
        'accuracy_results': accuracy_results,
        'predictor_performance': {
            'logistic_regression': {
                'accuracy': float(predictor_results['logistic']['accuracy']),
                'top_features': sorted(
                    predictor_results['logistic']['feature_importance'].items(),
                    key=lambda x: abs(x[1]), reverse=True
                )[:5]
            },
            'random_forest': {
                'accuracy': float(predictor_results['random_forest']['accuracy']),
                'top_features': sorted(
                    predictor_results['random_forest']['feature_importance'].items(),
                    key=lambda x: x[1], reverse=True
                )[:5]
            }
        }
    }
    
    summary_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Analysis summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze impact of 32B assistance on 7B performance')
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser.add_argument('--base', default=os.path.join(project_root, 'saves/base.jsonl'),
                       help='Path to base (7B only) results')
    parser.add_argument('--enhanced', default=os.path.join(project_root, 'saves/enhanced1000.jsonl'),
                       help='Path to enhanced (32B+7B) results')
    parser.add_argument('--output-dir', default=os.path.join(project_root, 'saves/assistance_impact_output'),
                       help='Output directory')
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                       help='Model to use for feature extraction')
    parser.add_argument('--n-tokens', type=int, default=1000,
                       help='Number of assistance tokens to analyze')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for filtering harmful assistance')
    
    args = parser.parse_args()
    
    # Load and match data (includes automatic evaluation)
    matched_data = load_and_match_results(args.base, args.enhanced, run_eval=True)
    
    # Categorize impacts
    categories = categorize_impacts(matched_data)
    
    # Initialize analyzer
    logger.info("Initializing token analyzer...")
    analyzer = TokenAnalyzer(args.model)
    
    # Extract features
    features, labels, indices = extract_features_for_prediction(
        analyzer, matched_data, args.n_tokens
    )
    
    # Train harm predictor
    predictor_results = train_harm_predictor(features, labels)
    
    # Calculate filtered accuracy
    accuracy_results = calculate_filtered_accuracy(
        categories, predictor_results, features, indices, args.threshold
    )
    
    # Update visualization with actual accuracies
    # Create visualizations
    create_visualizations(categories, predictor_results, features, labels, args.output_dir)
    
    # Save results
    save_results(categories, predictor_results, accuracy_results, args.output_dir)
    
    logger.info(f"\nAnalysis complete! Results saved to {args.output_dir}")
    
    # Print final recommendation
    improvement = accuracy_results['filtered_accuracy'] - accuracy_results['base_accuracy']
    logger.info(f"\n=== RECOMMENDATION ===")
    logger.info(f"Using selective 32B assistance improves accuracy by {improvement:.1%}")
    logger.info(f"From {accuracy_results['base_accuracy']:.1%} → {accuracy_results['filtered_accuracy']:.1%}")


if __name__ == "__main__":
    main()