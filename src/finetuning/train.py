"""Fine-tuning (LoRA) de um LLM com dados médicos internos do Hospital Vida Plena.

Uso típico (Colab, GPU T4):
    python -m src.finetuning.train

Uso para teste rápido local (modelo minúsculo, CPU, poucos passos):
    python -m src.finetuning.train --model sshleifer/tiny-gpt2 --epochs 1 --smoke-test

O modelo padrão (TinyLlama-1.1B-Chat) é aberto (sem gating no Hugging Face) e
roda com LoRA tanto em GPU CUDA (Colab) quanto em Apple Silicon (MPS) ou CPU.
Para um hospital real, trocar por um modelo maior (ex.: Llama 3, Falcon) e por
um dataset de dados internos muito maior do que a amostra didática deste repo.
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET = os.path.join(BASE_DIR, "data", "processed", "finetuning_dataset.jsonl")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "models", "llm-medico-lora")

PROMPT_TEMPLATE = (
    "Você é um assistente virtual médico do Hospital Vida Plena. "
    "Responda com base nos protocolos internos do hospital.\n\n"
    "### Pergunta:\n{instruction}\n\n### Resposta:\n{response}"
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_jsonl_dataset(path: str) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_and_tokenize(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    def _format(example):
        text = PROMPT_TEMPLATE.format(
            instruction=example["instruction"], response=example["response"]
        )
        return {"text": text}

    dataset = dataset.map(_format)

    def _tokenize(example):
        tokenized = tokenizer(
            example["text"], truncation=True, max_length=max_length, padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(_tokenize, remove_columns=dataset.column_names)


def build_lora_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA do assistente médico")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Limita passos/exemplos para validar rapidamente que o pipeline roda.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Dispositivo selecionado: {device}")

    dataset = load_jsonl_dataset(args.dataset)
    if args.smoke_test:
        dataset = dataset.select(range(min(4, len(dataset))))
    print(f"Exemplos de treino: {len(dataset)}")

    model, tokenizer = build_lora_model(args.model, device)
    tokenized_dataset = format_and_tokenize(dataset, tokenizer)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1 if args.smoke_test else args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=(device == "cpu"),
        max_steps=2 if args.smoke_test else -1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adaptador LoRA salvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
