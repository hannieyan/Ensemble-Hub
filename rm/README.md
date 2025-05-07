### Get the correct answer on the training set

Use LLaMA-Factory to validate the training set across different datasets and models:

- `DeepSeek-R1-Distill-Qwen-7B` on `gsm8k`

    ```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python scripts/vllm_infer.py \
      --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
      --template deepseek3 \
      --dataset gsm8k_train \
      --save_name output/gsm8k/train/deepseek-r1-7b-generated-predictions.jsonl
    ```

- `Qwen/Qwen2.5-Math-1.5B-Instruct` on `gsm8k`

    ```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python scripts/vllm_infer.py \
      --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
      --template qwen \
      --dataset gsm8k_train \
      --save_name output/gsm8k/train/qwen2.5-math-1.5b-generated-predictions.jsonl
    ```

- `DeepSeek-R1-Distill-Qwen-7B` on `hendrycks_math`

    ```shell
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python scripts/vllm_infer.py \
      --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
      --template deepseek3 \
      --dataset hendrycks_math_train \
      --save_name output/hendrycks_math/train/deepseek-r1-7b-generated-predictions.jsonl
    ```
  
- `Qwen/Qwen2.5-Math-1.5B-Instruct` on `hendrycks_math`

    ```shell
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python scripts/vllm_infer.py \
      --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
      --template qwen \
      --dataset hendrycks_math_train \
      --save_name output/hendrycks_math/train/qwen2.5-math-1.5b-generated-predictions.jsonl
    ```

...

Count the length of the answer:

```shell
python scripts/eval_len.py output/gsm8k/train/deepseek-r1-7b-generated-predictions.jsonl  --model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```


```shell
python scripts/eval_len.py output/hendrycks_math/train/deepseek-r1-7b-generated-predictions.jsonl  --model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

Eval Acc:

```shell
python scripts/eval_acc.py output/gsm8k/train/deepseek-r1-7b-generated-predictions.jsonl
```

```shell
python scripts/eval_acc.py output/hendrycks_math/train/deepseek-r1-7b-generated-predictions.jsonl
```

### Qwen/Qwen2.5-Math-1.5B-Instruct generation

Use Qwen/Qwen2.5-Math-1.5B-Instruct to continue reasoning about the truncated correct answer





### Prepare dataset for reward model sft





```
llamafactory-cli api --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct  --reward_model saves/qwen2.5-7b/lora/reward  --template qwen  --stage rm
```



```
curl -X POST -H "Content-Type: application/json" -d '{"model": "your_reward_model_id", "messages": ["一段文本", "第二段文本"]}' http://localhost:8000/v1/score/evaluation
```



