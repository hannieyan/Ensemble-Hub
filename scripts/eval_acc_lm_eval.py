import json
import logging
import time
import re
import fire
import os
from typing import Dict, List, Tuple

import sys
sys.path.append(os.path.dirname(__file__))
from grader import grade_answer


def extract_boxed_content(text: str) -> str:
    """Extract content inside \boxed{} command, handling nested braces."""
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


def compute_accuracy_from_record(record: Dict) -> Tuple[bool, str, str]:
    """
    Compute accuracy for a single record by comparing solution and response.
    
    Args:
        record: A dictionary containing 'solution' and 'resps' fields
        
    Returns:
        Tuple of (is_correct, predicted_boxed, reference_boxed)
    """
    # Extract reference answer from solution
    solution = record.get('doc', {}).get('solution', '')
    reference_boxed = extract_boxed_content(solution)
    
    # Extract predicted answer from resps
    resps = record.get('resps', [['']])
    if resps and resps[0]:
        response_text = resps[0][0]
    else:
        response_text = ''
    
    predicted_boxed = extract_boxed_content(response_text)
    
    # Grade the answer
    is_correct = grade_answer(predicted_boxed, reference_boxed) if predicted_boxed and reference_boxed else False
    
    return is_correct, predicted_boxed, reference_boxed


def generate_output_paths(input_path: str) -> Tuple[str, str]:
    """Generate output file paths based on input path."""
    base, ext = os.path.splitext(input_path)
    detailed_path = f"{base}-lm-eval-detailed-results.jsonl"
    score_path = f"{base}-lm-eval-score.json"
    return detailed_path, score_path


def main(filename: str):
    """
    Evaluate accuracy by comparing boxed content in solution and response.
    
    Args:
        filename: Path to the JSONL file containing model outputs
    """
    start_time = time.time()
    
    # Load and process data
    results = []
    total_correct = 0
    total_samples = 0
    
    with open(filename, 'r', encoding='utf-8') as f:
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
                
                # Include problem text if available
                if 'doc' in record and 'problem' in record['doc']:
                    result['problem'] = record['doc']['problem']
                
                results.append(result)
                total_samples += 1
                if is_correct:
                    total_correct += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num + 1}: {e}")
                continue
    
    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    
    # Prepare summary statistics
    summary = {
        "total_samples": total_samples,
        "total_correct": total_correct,
        "accuracy": round(overall_accuracy, 4),
        "accuracy_percentage": f"{overall_accuracy:.2f}%"
    }
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Total samples: {total_samples}")
    print(f"Correct answers: {total_correct}")
    print(f"Accuracy: {overall_accuracy:.4f}% ({total_correct}/{total_samples})")
    
    # Save results
    detailed_path, score_path = generate_output_paths(filename)
    
    # Save detailed results
    with open(detailed_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Save summary score
    with open(score_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"\nDone in {time.time() - start_time:.3f}s.")
    print(f"Score file saved to: {score_path}")
    print(f"Detailed results saved to: {detailed_path}")
    
    # Print some example errors for debugging
    print("\nExample errors (first 5):")
    error_count = 0
    for result in results:
        if not result['correct'] and error_count < 5:
            print(f"\nDoc ID {result['doc_id']}:")
            if 'problem' in result:
                print(f"Problem: {result['problem'][:100]}...")
            print(f"Expected: {result['reference_boxed']}")
            print(f"Got: {result['predicted_boxed']}")
            error_count += 1


if __name__ == "__main__":
    fire.Fire(main)