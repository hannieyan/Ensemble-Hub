#!/usr/bin/env python3
"""
Correctness Prediction Analysis Script
Analyzes whether PPL and entropy from a 7B model can predict answer correctness.
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

from complete_token_analysis import TokenAnalyzer, load_and_match_data


def extract_features_from_responses(analyzer, data, n_tokens=100, include_problem=True, max_samples=None):
    """Extract PPL and entropy features from responses."""
    features = []
    labels = []
    indices = []
    
    # Limit samples if specified
    if max_samples and len(data) > max_samples:
        logger.info(f"Limiting analysis to {max_samples} samples (out of {len(data)})")
        data = data[:max_samples]
    
    logger.info(f"Extracting features from {len(data)} responses...")
    
    for idx, item in enumerate(tqdm(data, desc="Processing responses")):
        try:
            # Get response text
            response = item['resps'][0][0] if item['resps'] else ""
            problem = item['doc']['problem']
            
            # Determine correctness
            is_correct = False
            if 'correct' in item:
                is_correct = item['correct']
            elif 'doc' in item and 'solution' in item['doc']:
                # Try to determine correctness by comparing with solution
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
                
                solution_boxed = extract_boxed_content(item['doc']['solution'])
                response_boxed = extract_boxed_content(response)
                
                if solution_boxed and response_boxed:
                    is_correct = grade_answer(solution_boxed, response_boxed)
            
            # Get token metrics using the appropriate method
            if include_problem:
                # Use the new method that analyzes only answer tokens but with problem context
                tokens, ppls, entropies = analyzer.compute_answer_token_metrics(problem, response, n_tokens)
            else:
                # Analyze only the response text
                tokens, ppls, entropies = analyzer.compute_token_metrics(response, n_tokens)
            
            if ppls and entropies:
                # Extract various features
                feature_dict = {
                    # PPL features
                    'ppl_mean': np.mean(ppls),
                    'ppl_std': np.std(ppls),
                    'ppl_median': np.median(ppls),
                    'ppl_max': np.max(ppls),
                    'ppl_min': np.min(ppls),
                    'ppl_q25': np.percentile(ppls, 25),
                    'ppl_q75': np.percentile(ppls, 75),
                    
                    # Entropy features
                    'entropy_mean': np.mean(entropies),
                    'entropy_std': np.std(entropies),
                    'entropy_median': np.median(entropies),
                    'entropy_max': np.max(entropies),
                    'entropy_min': np.min(entropies),
                    'entropy_q25': np.percentile(entropies, 25),
                    'entropy_q75': np.percentile(entropies, 75),
                    
                    # Trend features (first 20 tokens vs last 20 tokens within n_tokens limit)
                    'ppl_trend': 0,
                    'entropy_trend': 0,
                    
                    # Response length
                    'response_length': len(response.split()),
                    'token_count': len(ppls)
                }
                
                # Calculate trends if enough tokens
                if len(ppls) >= 40:
                    feature_dict['ppl_trend'] = np.mean(ppls[-20:]) - np.mean(ppls[:20])
                    feature_dict['entropy_trend'] = np.mean(entropies[-20:]) - np.mean(entropies[:20])
                
                features.append(feature_dict)
                labels.append(1 if is_correct else 0)
                indices.append(idx)
                
        except Exception as e:
            logger.debug(f"Error processing item {idx}: {e}")
            continue
    
    logger.info(f"Extracted features from {len(features)} responses")
    logger.info(f"Correct: {sum(labels)}, Incorrect: {len(labels) - sum(labels)}")
    
    return features, labels, indices


def train_prediction_models(features, labels):
    """Train various models to predict correctness."""
    # Convert to numpy arrays
    feature_names = list(features[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in features])
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 1. Simple threshold-based predictor
    logger.info("\n1. Threshold-based Predictor (PPL mean)")
    ppl_means = [f['ppl_mean'] for f in features]
    threshold_search = np.linspace(np.min(ppl_means), np.max(ppl_means), 100)
    best_threshold = 0
    best_accuracy = 0
    
    for thresh in threshold_search:
        predictions = [1 if ppl < thresh else 0 for ppl in ppl_means]
        accuracy = np.mean(np.array(predictions) == y)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh
    
    # Apply best threshold to test set
    test_ppl_means = X_test[:, feature_names.index('ppl_mean')]
    threshold_predictions = (test_ppl_means < best_threshold).astype(int)
    
    results['threshold'] = {
        'model': f'PPL < {best_threshold:.2f}',
        'accuracy': np.mean(threshold_predictions == y_test),
        'predictions': threshold_predictions,
        'threshold': best_threshold
    }
    
    logger.info(f"Best threshold: PPL < {best_threshold:.2f}")
    logger.info(f"Test accuracy: {results['threshold']['accuracy']:.3f}")
    
    # 2. Logistic Regression
    logger.info("\n2. Logistic Regression")
    lr = LogisticRegression(random_state=42, max_iter=1000)
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
    
    logger.info(f"Test accuracy: {results['logistic']['accuracy']:.3f}")
    logger.info("Top 5 features (by coefficient magnitude):")
    sorted_features = sorted(results['logistic']['feature_importance'].items(), 
                           key=lambda x: abs(x[1]), reverse=True)[:5]
    for feat, coef in sorted_features:
        logger.info(f"  {feat}: {coef:.3f}")
    
    # 3. Random Forest
    logger.info("\n3. Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
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
    
    logger.info(f"Test accuracy: {results['random_forest']['accuracy']:.3f}")
    logger.info("Top 5 features (by importance):")
    sorted_features = sorted(results['random_forest']['feature_importance'].items(), 
                           key=lambda x: x[1], reverse=True)[:5]
    for feat, imp in sorted_features:
        logger.info(f"  {feat}: {imp:.3f}")
    
    # Cross-validation scores
    logger.info("\n4. Cross-validation scores (5-fold)")
    cv_scores_lr = cross_val_score(lr, X_train_scaled, y_train, cv=5)
    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5)
    
    logger.info(f"Logistic Regression: {cv_scores_lr.mean():.3f} (+/- {cv_scores_lr.std() * 2:.3f})")
    logger.info(f"Random Forest: {cv_scores_rf.mean():.3f} (+/- {cv_scores_rf.std() * 2:.3f})")
    
    return results, X_test, y_test, scaler, feature_names


def create_prediction_visualizations(results, features, labels, output_dir):
    """Create visualizations for prediction analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Feature distributions by correctness
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PPL distribution
    ax1 = axes[0, 0]
    correct_ppls = [f['ppl_mean'] for f, l in zip(features, labels) if l == 1]
    incorrect_ppls = [f['ppl_mean'] for f, l in zip(features, labels) if l == 0]
    
    ax1.hist(correct_ppls, bins=30, alpha=0.6, label='Correct', color='green', density=True)
    ax1.hist(incorrect_ppls, bins=30, alpha=0.6, label='Incorrect', color='red', density=True)
    ax1.set_xlabel('Average PPL')
    ax1.set_ylabel('Density')
    ax1.set_title('PPL Distribution by Correctness')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Add threshold line if available
    if 'threshold' in results:
        ax1.axvline(results['threshold']['threshold'], color='black', linestyle='--', 
                   label=f'Threshold: {results["threshold"]["threshold"]:.1f}')
    
    # Entropy distribution
    ax2 = axes[0, 1]
    correct_entropy = [f['entropy_mean'] for f, l in zip(features, labels) if l == 1]
    incorrect_entropy = [f['entropy_mean'] for f, l in zip(features, labels) if l == 0]
    
    ax2.hist(correct_entropy, bins=30, alpha=0.6, label='Correct', color='green', density=True)
    ax2.hist(incorrect_entropy, bins=30, alpha=0.6, label='Incorrect', color='red', density=True)
    ax2.set_xlabel('Average Entropy')
    ax2.set_ylabel('Density')
    ax2.set_title('Entropy Distribution by Correctness')
    ax2.legend()
    
    # PPL vs Entropy scatter
    ax3 = axes[1, 0]
    for label, color, name in [(1, 'green', 'Correct'), (0, 'red', 'Incorrect')]:
        mask = np.array(labels) == label
        ppls = [f['ppl_mean'] for f, m in zip(features, mask) if m]
        entropies = [f['entropy_mean'] for f, m in zip(features, mask) if m]
        ax3.scatter(ppls, entropies, alpha=0.5, c=color, label=name, s=20)
    
    ax3.set_xlabel('Average PPL')
    ax3.set_ylabel('Average Entropy')
    ax3.set_title('PPL vs Entropy by Correctness')
    ax3.legend()
    ax3.set_xscale('log')
    
    # Model comparison
    ax4 = axes[1, 1]
    model_names = ['Threshold', 'Logistic Reg.', 'Random Forest']
    accuracies = [
        results['threshold']['accuracy'],
        results['logistic']['accuracy'],
        results['random_forest']['accuracy']
    ]
    
    bars = ax4.bar(model_names, accuracies, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax4.set_ylabel('Test Accuracy')
    ax4.set_title('Model Performance Comparison')
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Correctness Prediction Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    overview_path = os.path.join(output_dir, 'correctness_prediction_overview.png')
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    logger.info(f"Overview plot saved to {overview_path}")
    
    # 2. ROC curves
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curves
    for name, result in [('Logistic Regression', results['logistic']), 
                        ('Random Forest', results['random_forest'])]:
        if 'probabilities' in result:
            fpr, tpr, _ = roc_curve(results['y_test'], result['probabilities'])
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall curves
    for name, result in [('Logistic Regression', results['logistic']), 
                        ('Random Forest', results['random_forest'])]:
        if 'probabilities' in result:
            precision, recall, _ = precision_recall_curve(results['y_test'], result['probabilities'])
            ax2.plot(recall, precision, label=name)
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Model Performance Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    curves_path = os.path.join(output_dir, 'performance_curves.png')
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    logger.info(f"Performance curves saved to {curves_path}")
    
    # 3. Feature importance plot
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Logistic regression coefficients
    lr_importance = results['logistic']['feature_importance']
    sorted_features = sorted(lr_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    features_lr, coefs = zip(*sorted_features)
    
    y_pos = np.arange(len(features_lr))
    ax1.barh(y_pos, coefs, color=['green' if c > 0 else 'red' for c in coefs])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features_lr)
    ax1.set_xlabel('Coefficient')
    ax1.set_title('Logistic Regression: Top 10 Features')
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Random forest importance
    rf_importance = results['random_forest']['feature_importance']
    sorted_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    features_rf, importances = zip(*sorted_features)
    
    y_pos = np.arange(len(features_rf))
    ax2.barh(y_pos, importances, color='lightblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features_rf)
    ax2.set_xlabel('Importance')
    ax2.set_title('Random Forest: Top 10 Features')
    
    plt.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    importance_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(importance_path, dpi=150, bbox_inches='tight')
    logger.info(f"Feature importance plot saved to {importance_path}")
    
    # 4. Confusion matrices
    fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, result) in enumerate([('Threshold', results['threshold']),
                                          ('Logistic Reg.', results['logistic']),
                                          ('Random Forest', results['random_forest'])]):
        cm = confusion_matrix(results['y_test'], result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    matrices_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(matrices_path, dpi=150, bbox_inches='tight')
    logger.info(f"Confusion matrices saved to {matrices_path}")


def save_prediction_results(results, features, labels, output_dir):
    """Save prediction results and analysis."""
    # Prepare results summary
    summary = {
        'total_samples': len(labels),
        'correct_samples': sum(labels),
        'incorrect_samples': len(labels) - sum(labels),
        'models': {}
    }
    
    # Add model results
    for model_name, result in results.items():
        if model_name not in ['y_test', 'X_test']:
            summary['models'][model_name] = {
                'accuracy': float(result['accuracy']),
                'parameters': {}
            }
            
            if model_name == 'threshold':
                summary['models'][model_name]['parameters']['threshold'] = float(result['threshold'])
            elif 'feature_importance' in result:
                # Top 5 most important features
                sorted_features = sorted(result['feature_importance'].items(), 
                                       key=lambda x: abs(x[1]), reverse=True)[:5]
                summary['models'][model_name]['top_features'] = [
                    {'name': feat, 'importance': float(imp)} for feat, imp in sorted_features
                ]
    
    # Save summary
    summary_path = os.path.join(output_dir, 'prediction_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Prediction summary saved to {summary_path}")
    
    # Save detailed feature data
    feature_data = []
    for idx, (feat, label) in enumerate(zip(features, labels)):
        entry = {
            'index': idx,
            'correct': bool(label),
            'features': {k: float(v) for k, v in feat.items()}
        }
        feature_data.append(entry)
    
    feature_path = os.path.join(output_dir, 'feature_data.jsonl')
    with jsonlines.open(feature_path, mode='w') as writer:
        for entry in feature_data:
            writer.write(entry)
    
    logger.info(f"Feature data saved to {feature_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze if PPL/entropy can predict answer correctness')
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser.add_argument('--data', default=os.path.join(project_root, 'saves/base.jsonl'),
                       help='Path to response data file')
    parser.add_argument('--output-dir', default=os.path.join(project_root, 'saves/correctness_prediction_output'), 
                       help='Output directory')
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                       help='Model to use for analysis')
    parser.add_argument('--n-tokens', type=int, default=100, 
                       help='Number of tokens to analyze')
    parser.add_argument('--response-only', action='store_true',
                       help='Analyze only response text, excluding problem')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    logger.info("Initializing token analyzer...")
    analyzer = TokenAnalyzer(args.model)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    with jsonlines.open(args.data) as reader:
        data = list(reader)
    
    # If we need to determine correctness, we might need to run evaluation first
    if not any('correct' in item for item in data):
        logger.info("Running evaluation to determine correctness...")
        # Import and use the evaluation function from complete_token_analysis
        from complete_token_analysis import run_evaluation
        import tempfile
        
        # Create temporary directory for evaluation results
        temp_dir = tempfile.mkdtemp()
        try:
            eval_results_path = run_evaluation(args.data, temp_dir)
            
            # Load evaluation results and merge with data
            eval_results = {}
            with jsonlines.open(eval_results_path) as reader:
                for item in reader:
                    eval_results[item['index']] = item['correct']
            
            # Add correctness to data
            for idx, item in enumerate(data):
                if idx in eval_results:
                    item['correct'] = eval_results[idx]
                    
        finally:
            # Clean up temporary directory and all files in it
            import shutil
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary evaluation files")
    
    # Extract features
    include_problem = not args.response_only
    features, labels, indices = extract_features_from_responses(
        analyzer, data, args.n_tokens, include_problem
    )
    
    if len(features) < 50:
        logger.error(f"Not enough samples for analysis: {len(features)}")
        return
    
    # Train prediction models
    results, X_test, y_test, scaler, feature_names = train_prediction_models(features, labels)
    
    # Store test data in results for visualization
    results['X_test'] = X_test
    results['y_test'] = y_test
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    create_prediction_visualizations(results, features, labels, args.output_dir)
    
    # Save results
    save_prediction_results(results, features, labels, args.output_dir)
    
    logger.info("\nCorrectness prediction analysis completed!")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()