
## Run benchmark with a single model with lm-evaluation-harness

0-shot (arc_challenge_chat)

```bash
# DeepSeek-R1-Distill-Qwen-1.5B
accelerate launch -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --tasks arc_challenge_chat --batch_size 8 --log_samples --output_path results

# DeepSeek-R1-Distill-Qwen-7B  
accelerate launch -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --tasks arc_challenge_chat --batch_size 8 --log_samples --output_path results

# DeepSeek-R1-Distill-Qwen-14B
accelerate launch -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tasks arc_challenge_chat --batch_size 8 --log_samples --output_path results

# DeepSeek-R1-Distill-Qwen-32B
accelerate launch -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tasks arc_challenge_chat --batch_size 8 --log_samples --output_path results

# Qwen3-4B
accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen3-4B --tasks arc_challenge_chat --batch_size 8 --log_samples --output_path results

# Qwen3-8B
accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen3-8B --tasks arc_challenge_chat --batch_size 8 --log_samples --output_path results

# Qwen3-14B
accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen3-14B --tasks arc_challenge_chat --batch_size 8 --log_samples --output_path results

# Qwen3-32B
accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen3-32B --tasks arc_challenge_chat --batch_size 8 --log_samples --output_path results
```

3-shot (bbh)

```bash
# DeepSeek-R1-Distill-Qwen-1.5B
accelerate launch -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --tasks bbh --batch_size 8 --log_samples --output_path results

# DeepSeek-R1-Distill-Qwen-7B
accelerate launch -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --tasks bbh --batch_size 8 --log_samples --output_path results

# DeepSeek-R1-Distill-Qwen-14B
accelerate launch -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tasks bbh --batch_size 8 --log_samples --output_path results

# DeepSeek-R1-Distill-Qwen-32B
accelerate launch -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tasks bbh --batch_size 8 --log_samples --output_path results

# Qwen3-4B
accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen3-4B --tasks bbh --batch_size 8 --log_samples --output_path results

# Qwen3-8B
accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen3-8B --tasks bbh --batch_size 8 --log_samples --output_path results

# Qwen3-14B
accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen3-14B --tasks bbh --batch_size 8 --log_samples --output_path results

# Qwen3-32B
accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen3-32B --tasks bbh --batch_size 8 --log_samples --output_path results
```

5-shot (gsm8k,mmlu_generative,triviaqa,nq_open)

```bash
# DeepSeek-R1-Distill-Qwen-1.5B
accelerate launch -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --tasks gsm8k,mmlu_generative,triviaqa,nq_open --batch_size 8 --num_fewshot 5 --log_samples --output_path results

# DeepSeek-R1-Distill-Qwen-7B
accelerate launch -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --tasks gsm8k,mmlu_generative,triviaqa,nq_open --batch_size 8 --num_fewshot 5 --log_samples --output_path results

# DeepSeek-R1-Distill-Qwen-14B
accelerate launch -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tasks gsm8k,mmlu_generative,triviaqa,nq_open --batch_size 8 --num_fewshot 5 --log_samples --output_path results

# DeepSeek-R1-Distill-Qwen-32B
accelerate launch -m lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tasks gsm8k,mmlu_generative,triviaqa,nq_open --batch_size 8 --num_fewshot 5 --log_samples --output_path results

# Qwen3-4B
accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen3-4B --tasks gsm8k,mmlu_generative,triviaqa,nq_open --batch_size 8 --num_fewshot 5 --log_samples --output_path results

# Qwen3-8B
accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen3-8B --tasks gsm8k,mmlu_generative,triviaqa,nq_open --batch_size 8 --num_fewshot 5 --log_samples --output_path results

# Qwen3-14B
accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen3-14B --tasks gsm8k,mmlu_generative,triviaqa,nq_open --batch_size 8 --num_fewshot 5 --log_samples --output_path results

# Qwen3-32B
accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen3-32B --tasks gsm8k,mmlu_generative,triviaqa,nq_open --batch_size 8 --num_fewshot 5 --log_samples --output_path results
```

