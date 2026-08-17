"""Preparação do dataset de fine-tuning: curadoria, anonimização e formatação.

Combina:
  1) Amostra de QA clínico no estilo MedQuAD/PubMedQA (real, se disponível
     localmente em data/raw/MedQuAD; caso contrário usa uma amostra embutida
     representativa, claramente sinalizada como fallback).
  2) Exemplos sintéticos gerados a partir dos protocolos internos do
     Hospital Vida Plena (data/raw/protocolos_hospitalares), simulando
     perguntas frequentes de médicos e respostas fundamentadas nos protocolos.

Saída: data/processed/finetuning_dataset.jsonl, no formato de instrução:
    {"instruction": ..., "context": ..., "response": ..., "source": ...}
"""

import glob
import json
import os
import re

try:
    from .anonymize import anonymize_text
except ImportError:  # executado como script solto (ex.: dentro do notebook Colab)
    from anonymize import anonymize_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PROTOCOLOS_DIR = os.path.join(RAW_DIR, "protocolos_hospitalares")
MEDQUAD_DIR = os.path.join(RAW_DIR, "MedQuAD")

# Amostra de fallback no estilo MedQuAD (perguntas e respostas clínicas gerais).
# Usada apenas se o dataset MedQuAD completo não tiver sido baixado em
# data/raw/MedQuAD (ver README para instruções de download).
MEDQUAD_FALLBACK_SAMPLE = [
    {
        "instruction": "O que é hipertensão arterial sistêmica?",
        "response": (
            "Hipertensão arterial sistêmica é uma condição clínica caracterizada "
            "por níveis elevados e sustentados de pressão arterial (geralmente "
            ">= 140x90 mmHg), associada a maior risco de eventos cardiovasculares "
            "e renais."
        ),
    },
    {
        "instruction": "Quais são os sinais de alarme na dengue?",
        "response": (
            "Dor abdominal intensa e contínua, vômitos persistentes, sangramento "
            "de mucosas, letargia ou irritabilidade, hepatomegalia e aumento "
            "progressivo do hematócrito com queda rápida de plaquetas."
        ),
    },
    {
        "instruction": "O que caracteriza um choque séptico?",
        "response": (
            "Choque séptico é a sepse associada a hipotensão persistente que "
            "requer vasopressor para manter pressão arterial média >= 65 mmHg, "
            "e lactato sérico > 2 mmol/L, mesmo após adequada reposição volêmica."
        ),
    },
    {
        "instruction": "Qual a diferença entre urgência e emergência hipertensiva?",
        "response": (
            "Na urgência hipertensiva há pressão arterial muito elevada sem lesão "
            "aguda de órgão-alvo; na emergência hipertensiva há lesão aguda de "
            "órgão-alvo associada, exigindo redução pressórica imediata e "
            "monitorada."
        ),
    },
    {
        "instruction": "Por que evitar metas glicêmicas muito rígidas em pacientes críticos?",
        "response": (
            "Metas glicêmicas excessivamente rígidas (por exemplo, < 110 mg/dL) "
            "aumentam significativamente o risco de hipoglicemia em pacientes "
            "críticos, sem benefício adicional comprovado de mortalidade."
        ),
    },
]

QA_PATTERN = re.compile(r"P:\s*(.+?)\nR:\s*(.+?)(?=\nP:|\Z)", re.DOTALL)


def load_medquad_sample(limit: int = 200) -> list[dict]:
    """Carrega amostra do MedQuAD real se disponível, senão usa fallback embutido."""
    if os.path.isdir(MEDQUAD_DIR):
        pairs = []
        xml_files = glob.glob(os.path.join(MEDQUAD_DIR, "**", "*.xml"), recursive=True)
        try:
            import xml.etree.ElementTree as ET

            for path in xml_files[:limit]:
                tree = ET.parse(path)
                root = tree.getroot()
                for qa_pair in root.iter("QAPair"):
                    question = qa_pair.findtext("Question")
                    answer = qa_pair.findtext("Answer")
                    if question and answer and answer.strip():
                        pairs.append({"instruction": question.strip(), "response": answer.strip()})
                if len(pairs) >= limit:
                    break
        except Exception as exc:
            print(f"Falha ao ler MedQuAD real ({exc}); usando amostra embutida.")
            return list(MEDQUAD_FALLBACK_SAMPLE)

        if pairs:
            print(f"MedQuAD real encontrado: {len(pairs)} pares carregados.")
            return pairs[:limit]

    print("MedQuAD real não encontrado em data/raw/MedQuAD — usando amostra embutida.")
    return list(MEDQUAD_FALLBACK_SAMPLE)


def load_hospital_faq() -> list[dict]:
    """Extrai pares de pergunta/resposta do FAQ interno do hospital."""
    faq_path = os.path.join(PROTOCOLOS_DIR, "faq_medicos.txt")
    if not os.path.exists(faq_path):
        return []

    with open(faq_path, encoding="utf-8") as f:
        content = f.read()

    pairs = []
    for match in QA_PATTERN.finditer(content):
        question, answer = match.group(1).strip(), match.group(2).strip()
        answer = " ".join(line.strip() for line in answer.splitlines())
        pairs.append({"instruction": question, "response": answer, "source": "faq_medicos.txt"})

    return pairs


def curate(records: list[dict], min_len: int = 15) -> list[dict]:
    """Remove duplicatas e respostas curtas demais para servir de bom exemplo."""
    seen = set()
    curated = []
    for r in records:
        key = r["instruction"].strip().lower()
        if key in seen:
            continue
        if len(r.get("response", "")) < min_len:
            continue
        seen.add(key)
        curated.append(r)
    return curated


def anonymize_dataset(records: list[dict]) -> list[dict]:
    return [
        {
            **r,
            "instruction": anonymize_text(r["instruction"]),
            "response": anonymize_text(r["response"]),
        }
        for r in records
    ]


def build_finetuning_dataset() -> str:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    medquad = load_medquad_sample()
    for r in medquad:
        r.setdefault("source", "MedQuAD")

    hospital_faq = load_hospital_faq()

    all_records = medquad + hospital_faq
    all_records = curate(all_records)
    all_records = anonymize_dataset(all_records)

    out_path = os.path.join(PROCESSED_DIR, "finetuning_dataset.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Dataset de fine-tuning gerado: {out_path} ({len(all_records)} exemplos)")
    return out_path


if __name__ == "__main__":
    build_finetuning_dataset()
