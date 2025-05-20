import json
import math
import re
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from grader import grade_answer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
GEN_KWARGS = {
    "max_new_tokens": 16384,
    "do_sample": True,
    "temperature": 0.95,
    "top_p": 0.7,
    "top_k": 50,
    "repetition_penalty": 1.0,
}

def extract_boxed_answers(text: str):
    answers = []
    idx = 0
    while True:
        start = text.find(r"\boxed{", idx)
        if start == -1:
            break
        brace_count = 0
        j = start
        while j < len(text) and text[j] != '{':
            j += 1
        if j >= len(text):
            break
        brace_count = 1
        j += 1
        content_start = j
        while j < len(text) and brace_count > 0:
            if text[j] == '{':
                brace_count += 1
            elif text[j] == '}':
                brace_count -= 1
            j += 1
        content_end = j - 1
        if brace_count == 0:
            content = text[content_start:content_end]
            answers.append(content.strip())
        else:
            break
        idx = j
    return answers

def process_chunk(gpu_id: int, data_slice: list):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()

    results = []
    batch_size = 1

    for i in range(0, len(data_slice), batch_size):
        batch_questions = data_slice[i : i + batch_size]
        prompts = []
        for item in batch_questions:
            prompt = item.get("input", "")
            if not prompt.endswith("}"):
                prompt += " "
            prompt += "Please reason step by step, and put your final answer within \\boxed{}."
            prompts.append(prompt)

        encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = encodings["input_ids"]
        attention_mask = encodings.get("attention_mask")

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **GEN_KWARGS,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )

        sequences = outputs.sequences
        input_lengths = (input_ids != tokenizer.pad_token_id).sum(dim=1)
        transition_scores = model.compute_transition_scores(
            sequences, outputs.scores, normalize_logits=True
        )

        for idx, seq in enumerate(sequences):
            seq = seq.tolist()
            prompt_len = int(input_lengths[idx].item())
            generated_ids = seq[prompt_len:]
            full_text = tokenizer.decode(seq, skip_special_tokens=True)

            # Generation token confidence
            g_logps = transition_scores[idx].cpu().tolist()
            g_confs = [math.exp(lp) for lp in g_logps]
            g_avg_conf = sum(g_confs)/len(g_confs) if g_confs else 0.0

            # Prompt perplexity and confidence
            prompt = batch_questions[idx]["input"]
            try:
                with torch.no_grad():
                    p_in = tokenizer(prompt, return_tensors="pt").to(device)
                    logits = model(**p_in).logits
                    ids = p_in["input_ids"]
                    probs = torch.softmax(logits[:, :-1, :], dim=-1)
                    p_token = probs.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
                    p_token = p_token.cpu().tolist()
                prompt_avg_conf = sum(p_token)/len(p_token)
                prompt_ppl = math.exp(-sum(math.log(max(p, 1e-12)) for p in p_token)/len(p_token))
            except Exception as e:
                print(f"[GPU {gpu_id}] failed to compute prompt PPL/conf for sample {i + idx + 1}: {e}")
                prompt_ppl = float("inf")
                prompt_avg_conf = 0.0

            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            reference_solution = batch_questions[idx]["output"]
            # print("[Predicted]:", generated_text)
            # print("[Label]:", reference_solution)
            
            pred_answers = extract_boxed_answers(generated_text)
            ref_answers = extract_boxed_answers(reference_solution)

            # print("[Boxed Pred]:", pred_answers)
            # print("[Boxed Ref ]:", ref_answers)

            # 判断逻辑
            is_correct = False
            for pred in pred_answers:
                if any(grade_answer(pred, ref) for ref in ref_answers):
                    is_correct = True
                    break


            result = {
                "input": prompt,
                "prompt_ppl": prompt_ppl,
                "prompt_confidence": prompt_avg_conf,
                "reference": reference_solution,
                "predicted": full_text,
                "generation_token_confidences": g_confs,
                "generation_avg_confidence": g_avg_conf,
                "is_correct": is_correct,
            }
            results.append(result)

            global_index = i + idx
            print(
                f"[GPU {gpu_id}] {global_index+1}/{len(data_slice)} | "
                f"correct={is_correct} | prompt_ppl={prompt_ppl:.2f} | "
                f"prompt_conf={prompt_avg_conf:.3f} | gen_conf={g_avg_conf:.3f}"
            )

    out_file = f"results_gpu{gpu_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    with open("math500.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # data = data[:32]
    
    total = len(data)
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("No GPU available for inference.")

    chunk_sizes = []
    base_chunk = total // num_gpus
    remainder = total % num_gpus
    for i in range(num_gpus):
        size = base_chunk + (1 if i < remainder else 0)
        chunk_sizes.append(size)

    processes = []
    start_index = 0
    for gpu_id, size in enumerate(chunk_sizes):
        if size <= 0:
            continue
        end_index = start_index + size
        data_slice = data[start_index:end_index]
        p = mp.Process(target=process_chunk, args=(gpu_id, data_slice))
        p.start()
        processes.append(p)
        start_index = end_index

    for p in processes:
        p.join()

    all_results = []
    for gpu_id, size in enumerate(chunk_sizes):
        if size <= 0:
            continue
        temp_file = f"results_gpu{gpu_id}.json"
        with open(temp_file, "r", encoding="utf-8") as f:
            part = json.load(f)
            all_results.extend(part)

    with open("math500_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
