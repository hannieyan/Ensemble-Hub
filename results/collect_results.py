#!/usr/bin/env python3
"""
Collect all evaluation results and generate a markdown table
"""
import json
import os
from pathlib import Path

def collect_results():
    """Collect all accuracy and token metrics results"""
    
    # Find all score files
    score_files = []
    token_files = []
    
    for root, dirs, files in os.walk("saves"):
        for file in files:
            if file.endswith("-lm-eval-score.json"):
                score_files.append(os.path.join(root, file))
            elif file.endswith("-token-metrics.json") and "detailed-results" not in file:
                token_files.append(os.path.join(root, file))
    
    # Parse results
    results = {}
    
    # Parse accuracy results
    for score_file in score_files:
        with open(score_file, 'r') as f:
            data = json.load(f)
        
        # Extract model and task info from path
        parts = score_file.split("/")
        model_name = parts[1].replace("__", "/")
        filename = parts[2]
        
        # Extract task name - handle hendrycks_math_algebra format
        if "hendrycks_math_" in filename:
            task_name = filename.split("hendrycks_math_")[1].split("_")[0]  # e.g., "algebra"
        else:
            task_name = filename.split("_")[2]  # fallback
        
        if model_name not in results:
            results[model_name] = {}
        
        if task_name not in results[model_name]:
            results[model_name][task_name] = {}
            
        results[model_name][task_name]["accuracy"] = data["accuracy"]
        results[model_name][task_name]["total_samples"] = data["total_samples"]
    
    # Parse token metrics results
    for token_file in token_files:
        with open(token_file, 'r') as f:
            data = json.load(f)
        
        # Extract model and task info from path
        parts = token_file.split("/")
        model_name = parts[1].replace("__", "/")
        filename = parts[2]
        
        # Extract task name - handle hendrycks_math_algebra format
        if "hendrycks_math_" in filename:
            task_name = filename.split("hendrycks_math_")[1].split("_")[0]  # e.g., "algebra"
        else:
            task_name = filename.split("_")[2]  # fallback
        
        if model_name not in results:
            results[model_name] = {}
        
        if task_name not in results[model_name]:
            results[model_name][task_name] = {}
            
        results[model_name][task_name]["avg_tokens"] = data["token_stats"]["average"]
        results[model_name][task_name]["avg_chars"] = data["char_stats"]["average"]
    
    return results

def generate_markdown_table(results):
    """Generate markdown table from results"""
    
    # Get all tasks
    all_tasks = set()
    for model_data in results.values():
        all_tasks.update(model_data.keys())
    
    all_tasks = sorted(all_tasks)
    
    # Generate table
    md_lines = []
    
    # Header
    md_lines.append("# DeepSeek-R1-Distill-Qwen Models - Math Evaluation Results")
    md_lines.append("")
    md_lines.append("## Overall Performance Summary")
    md_lines.append("")
    
    # Create summary table
    md_lines.append("| Model | Overall Accuracy | Avg Response Length (tokens) | Avg Response Length (chars) |")
    md_lines.append("|-------|------------------|------------------------------|----------------------------|")
    
    for model_name in sorted(results.keys()):
        total_samples = 0
        total_correct = 0
        total_tokens = 0
        total_chars = 0
        task_count = 0
        
        for task_name in all_tasks:
            if task_name in results[model_name]:
                task_data = results[model_name][task_name]
                if "total_samples" in task_data and "accuracy" in task_data:
                    samples = task_data["total_samples"]
                    acc = task_data["accuracy"]
                    total_samples += samples
                    total_correct += int(samples * acc / 100)
                    
                if "avg_tokens" in task_data:
                    total_tokens += task_data["avg_tokens"]
                    task_count += 1
                    
                if "avg_chars" in task_data:
                    total_chars += task_data["avg_chars"]
        
        overall_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0
        avg_tokens = total_tokens / task_count if task_count > 0 else 0
        avg_chars = total_chars / task_count if task_count > 0 else 0
        
        md_lines.append(f"| {model_name} | {overall_acc:.2f}% | {avg_tokens:.1f} | {avg_chars:.1f} |")
    
    md_lines.append("")
    md_lines.append("## Detailed Results by Task")
    md_lines.append("")
    
    # Detailed table
    header = "| Model | Task | Accuracy | Samples | Avg Tokens | Avg Chars |"
    separator = "|-------|------|----------|---------|------------|-----------|"
    md_lines.append(header)
    md_lines.append(separator)
    
    for model_name in sorted(results.keys()):
        for task_name in all_tasks:
            if task_name in results[model_name]:
                task_data = results[model_name][task_name]
                acc = task_data.get("accuracy", 0)
                samples = task_data.get("total_samples", 0)
                tokens = task_data.get("avg_tokens", 0)
                chars = task_data.get("avg_chars", 0)
                
                md_lines.append(f"| {model_name} | {task_name} | {acc:.2f}% | {samples} | {tokens:.1f} | {chars:.1f} |")
    
    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("- **Accuracy**: Percentage of correct answers based on boxed content matching")
    md_lines.append("- **Avg Tokens**: Average number of tokens per response")
    md_lines.append("- **Avg Chars**: Average number of characters per response")
    md_lines.append("- **Tasks**: algebra, counting_and_prob, geometry, intermediate_algebra, num_theory, prealgebra, precalc")
    
    return "\n".join(md_lines)

def main():
    print("Collecting evaluation results...")
    results = collect_results()
    
    print("Generating markdown table...")
    markdown_table = generate_markdown_table(results)
    
    # Save to file
    output_file = "single_model_results.md"
    with open(output_file, 'w') as f:
        f.write(markdown_table)
    
    print(f"Results saved to {output_file}")
    print("\nMarkdown table:")
    print(markdown_table)

if __name__ == "__main__":
    main()