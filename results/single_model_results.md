# DeepSeek-R1-Distill-Qwen Models - Math Evaluation Results

## Overall Performance Summary

| Model | Overall Accuracy | Avg Response Length (tokens) | Avg Response Length (chars) |
|-------|------------------|------------------------------|----------------------------|
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 65.18% | 4315.8 | 12096.0 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | 86.86% | 2944.8 | 8141.1 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | 88.34% | 2849.3 | 7886.6 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 84.38% | 3198.8 | 8822.7 |

## Detailed Results by Task

| Model | Task | Accuracy | Samples | Avg Tokens | Avg Chars |
|-------|------|----------|---------|------------|-----------|
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | algebra | 84.33% | 1187 | 2782.3 | 7642.9 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | counting | 53.38% | 474 | 4839.9 | 15211.9 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | geometry | 56.16% | 479 | 4757.5 | 14385.5 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | intermediate | 52.60% | 903 | 5105.7 | 13171.7 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | num | 59.44% | 540 | 4657.3 | 11771.9 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | prealgebra | 74.86% | 871 | 3124.4 | 9260.4 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | precalc | 53.48% | 546 | 4943.3 | 13227.4 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | algebra | 95.79% | 1187 | 1893.6 | 5123.8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | counting | 83.97% | 474 | 2971.4 | 9241.7 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | geometry | 80.17% | 479 | 3356.9 | 9940.6 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | intermediate | 80.07% | 903 | 3958.0 | 10037.0 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | num | 87.41% | 540 | 3041.7 | 7826.8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | prealgebra | 91.50% | 871 | 1708.6 | 5085.3 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | precalc | 80.22% | 546 | 3683.6 | 9732.1 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | algebra | 96.63% | 1187 | 1834.3 | 5005.6 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | counting | 88.19% | 474 | 2758.5 | 8583.9 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | geometry | 81.42% | 479 | 3499.5 | 10354.5 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | intermediate | 81.51% | 903 | 3821.7 | 9649.2 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | num | 89.07% | 540 | 2816.7 | 7265.2 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | prealgebra | 92.31% | 871 | 1604.4 | 4808.7 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | precalc | 81.14% | 546 | 3610.2 | 9539.4 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | algebra | 95.20% | 1187 | 2067.1 | 5618.9 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | counting | 81.22% | 474 | 3348.7 | 10385.8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | geometry | 75.99% | 479 | 3628.3 | 10787.2 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | intermediate | 76.19% | 903 | 4153.3 | 10531.4 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | num | 82.04% | 540 | 3432.0 | 8641.6 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | prealgebra | 90.59% | 871 | 1926.8 | 5753.2 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | precalc | 77.66% | 546 | 3835.2 | 10040.9 |

## Combined Results (Accuracy % & Avg Tokens)

| Model | Metric | algebra | counting | geometry | intermediate | num | prealgebra | precalc |
|-------|--------|---------|----------|----------|--------------|-----|------------|---------|
| **DeepSeek-R1-Distill-Qwen-1.5B** | Accuracy | 84.33 | 53.38 | 56.16 | 52.60 | 59.44 | 74.86 | 53.48 |
| **DeepSeek-R1-Distill-Qwen-1.5B** | Avg Tokens | 2782.3 | 4839.9 | 4757.5 | 5105.7 | 4657.3 | 3124.4 | 4943.3 |
| **DeepSeek-R1-Distill-Qwen-7B** | Accuracy | 95.20 | 81.22 | 75.99 | 76.19 | 82.04 | 90.59 | 77.66 |
| **DeepSeek-R1-Distill-Qwen-7B** | Avg Tokens | 2067.1 | 3348.7 | 3628.3 | 4153.3 | 3432.0 | 1926.8 | 3835.2 |
| **DeepSeek-R1-Distill-Qwen-14B** | Accuracy | 95.79 | 83.97 | 80.17 | 80.07 | 87.41 | 91.50 | 80.22 |
| **DeepSeek-R1-Distill-Qwen-14B** | Avg Tokens | 1893.6 | 2971.4 | 3356.9 | 3958.0 | 3041.7 | 1708.6 | 3683.6 |
| **DeepSeek-R1-Distill-Qwen-32B** | Accuracy | 96.63 | 88.19 | 81.42 | 81.51 | 89.07 | 92.31 | 81.14 |
| **DeepSeek-R1-Distill-Qwen-32B** | Avg Tokens | 1834.3 | 2758.5 | 3499.5 | 3821.7 | 2816.7 | 1604.4 | 3610.2 |

## Notes
- **Accuracy**: Percentage of correct answers based on boxed content matching
- **Avg Tokens**: Average number of tokens per response
- **Avg Chars**: Average number of characters per response
- **Tasks**: algebra, counting_and_prob, geometry, intermediate_algebra, num_theory, prealgebra, precalc