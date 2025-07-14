import json
import time
import fire
import os
from typing import Dict, List, Tuple
from transformers import AutoTokenizer


def compute_response_token_count(record: Dict, tokenizer) -> Dict:
    """Compute token count for the response in a single record."""
    # Extract response text from resps field
    resps = record.get('resps', [['']])
    if resps and resps[0]:
        response_text = resps[0][0]
    else:
        response_text = ''
    
    # Count tokens using the tokenizer
    tokens = tokenizer.encode(response_text, add_special_tokens=False)
    token_count = len(tokens)
    
    return {
        "response_text": response_text,
        "token_count": token_count,
        "char_count": len(response_text)
    }


def generate_output_paths(input_path: str) -> Tuple[str, str, str]:
    """Generate output file paths based on input path."""
    base, ext = os.path.splitext(input_path)
    metrics_path = f"{base}-token-metrics.json"
    distribution_path = f"{base}-token-distribution.json"
    detailed_path = f"{base}-token-details.jsonl"
    return metrics_path, distribution_path, detailed_path


def main(filename: str, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
    """
    Calculate average token count for responses in lm-eval format JSONL files.
    
    Args:
        filename: Path to the JSONL file with model outputs
        model_name: Name of the tokenizer to use (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
    """
    start_time = time.time()
    print(f"Loading tokenizer: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to Qwen tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
    
    print(f"Processing file: {filename}")
    
    # Process data
    token_counts = []
    char_counts = []
    detailed_results = []
    total_samples = 0
    empty_responses = 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
                
            try:
                record = json.loads(line)
                result = compute_response_token_count(record, tokenizer)
                
                token_count = result["token_count"]
                char_count = result["char_count"]
                
                token_counts.append(token_count)
                char_counts.append(char_count)
                
                if token_count == 0:
                    empty_responses += 1
                
                # Add detailed result
                detailed_result = {
                    "index": line_num,
                    "doc_id": record.get("doc_id", line_num),
                    "token_count": token_count,
                    "char_count": char_count,
                    "response_preview": result["response_text"][:100] + "..." if len(result["response_text"]) > 100 else result["response_text"]
                }
                
                # Include problem if available
                if 'doc' in record and 'problem' in record['doc']:
                    detailed_result['problem'] = record['doc']['problem'][:100] + "..." if len(record['doc']['problem']) > 100 else record['doc']['problem']
                
                detailed_results.append(detailed_result)
                total_samples += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num + 1}: {e}")
                continue
    
    # Calculate statistics
    if total_samples > 0:
        avg_token_count = sum(token_counts) / total_samples
        avg_char_count = sum(char_counts) / total_samples
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        
        # Calculate percentiles
        sorted_tokens = sorted(token_counts)
        p25_idx = int(len(sorted_tokens) * 0.25)
        p50_idx = int(len(sorted_tokens) * 0.50)
        p75_idx = int(len(sorted_tokens) * 0.75)
        
        p25_tokens = sorted_tokens[p25_idx]
        p50_tokens = sorted_tokens[p50_idx]
        p75_tokens = sorted_tokens[p75_idx]
    else:
        avg_token_count = avg_char_count = 0
        min_tokens = max_tokens = 0
        p25_tokens = p50_tokens = p75_tokens = 0
    
    # Print summary
    print("\n=== Response Length Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Empty responses: {empty_responses} ({empty_responses/total_samples*100:.1f}%)" if total_samples > 0 else "Empty responses: 0")
    print(f"\nToken Statistics:")
    print(f"  Average tokens per response: {avg_token_count:.2f}")
    print(f"  Min tokens: {min_tokens}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  25th percentile: {p25_tokens}")
    print(f"  50th percentile (median): {p50_tokens}")
    print(f"  75th percentile: {p75_tokens}")
    print(f"\nCharacter Statistics:")
    print(f"  Average characters per response: {avg_char_count:.2f}")
    
    # Save metrics
    metrics = {
        "total_samples": total_samples,
        "empty_responses": empty_responses,
        "empty_response_rate": round(empty_responses/total_samples*100, 2) if total_samples > 0 else 0,
        "token_stats": {
            "total_tokens": sum(token_counts),
            "average": round(avg_token_count, 2),
            "min": min_tokens,
            "max": max_tokens,
            "p25": p25_tokens,
            "p50_median": p50_tokens,
            "p75": p75_tokens
        },
        "char_stats": {
            "total_chars": sum(char_counts),
            "average": round(avg_char_count, 2)
        }
    }
    
    # Save outputs
    metrics_path, distribution_path, detailed_path = generate_output_paths(filename)
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    with open(distribution_path, 'w', encoding='utf-8') as f:
        json.dump(token_counts, f, indent=4)
    
    with open(detailed_path, 'w', encoding='utf-8') as f:
        for result in detailed_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\nDone in {time.time() - start_time:.3f}s.")
    print(f"Token metrics saved to: {metrics_path}")
    print(f"Token distribution saved to: {distribution_path}")
    print(f"Detailed results saved to: {detailed_path}")
    
    # Show some examples of longest responses
    print("\nExamples of longest responses:")
    sorted_results = sorted(detailed_results, key=lambda x: x['token_count'], reverse=True)
    for i, result in enumerate(sorted_results[:3]):
        print(f"\n{i+1}. Doc ID {result['doc_id']} ({result['token_count']} tokens):")
        if 'problem' in result:
            print(f"   Problem: {result['problem']}")
        print(f"   Response preview: {result['response_preview']}")


if __name__ == "__main__":
    fire.Fire(main)