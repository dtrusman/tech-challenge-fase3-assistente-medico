"""Avaliação do modelo fine-tuned: compara respostas do modelo base vs. modelo
ajustado (LoRA) em perguntas de validação, usando ROUGE-L como métrica de
sobreposição textual com a resposta de referência dos protocolos internos.
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .train import PROMPT_TEMPLATE, get_device, load_jsonl_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET = os.path.join(BASE_DIR, "data", "processed", "finetuning_dataset.jsonl")
DEFAULT_ADAPTER_DIR = os.path.join(BASE_DIR, "models", "llm-medico-lora")
DEFAULT_REPORT_PATH = os.path.join(BASE_DIR, "reports", "avaliacao_modelo.json")


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    """ROUGE-L (F1) simples baseado em maior subsequência comum, sem dependências extras."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0

    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]

    precision = lcs / n
    recall = lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def generate(model, tokenizer, instruction: str, device: str, max_new_tokens: int = 120) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction, response="")
    prompt = prompt.split("### Resposta:")[0] + "### Resposta:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text.split("### Resposta:")[-1].strip()


def evaluate(
    base_model_name: str,
    adapter_dir: str,
    dataset_path: str,
    n_samples: int = 5,
    report_path: str = DEFAULT_REPORT_PATH,
) -> dict:
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
    finetuned_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_model_name).to(device), adapter_dir
    )

    dataset = load_jsonl_dataset(dataset_path)
    n_samples = min(n_samples, len(dataset))
    sample = dataset.select(range(n_samples))

    results = []
    for example in sample:
        instruction = example["instruction"]
        reference = example["response"]

        base_answer = generate(base_model, tokenizer, instruction, device)
        finetuned_answer = generate(finetuned_model, tokenizer, instruction, device)

        results.append(
            {
                "instruction": instruction,
                "reference": reference,
                "base_model_answer": base_answer,
                "finetuned_model_answer": finetuned_answer,
                "rouge_l_base": round(rouge_l_f1(reference, base_answer), 4),
                "rouge_l_finetuned": round(rouge_l_f1(reference, finetuned_answer), 4),
            }
        )

    avg_base = sum(r["rouge_l_base"] for r in results) / len(results)
    avg_finetuned = sum(r["rouge_l_finetuned"] for r in results) / len(results)

    report = {
        "base_model": base_model_name,
        "adapter_dir": adapter_dir,
        "n_samples": n_samples,
        "avg_rouge_l_base": round(avg_base, 4),
        "avg_rouge_l_finetuned": round(avg_finetuned, 4),
        "samples": results,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"ROUGE-L médio — base: {avg_base:.4f} | fine-tuned: {avg_finetuned:.4f}")
    print(f"Relatório salvo em: {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação do modelo fine-tuned vs. base")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--n-samples", type=int, default=5)
    args = parser.parse_args()

    evaluate(args.model, args.adapter_dir, args.dataset, args.n_samples)
