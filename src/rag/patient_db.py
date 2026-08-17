"""Acesso à base estruturada de prontuários (mock) do Hospital Vida Plena.

Em produção, este módulo seria substituído por uma consulta real ao sistema
de prontuário eletrônico (HL7/FHIR, banco relacional, etc.). Aqui usamos um
JSON local para demonstrar a integração exigida pelo desafio: contextualizar
as respostas do assistente com dados atualizados do paciente.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRONTUARIOS_PATH = os.path.join(BASE_DIR, "data", "processed", "prontuarios_mock.json")


def load_prontuarios(path: str = PRONTUARIOS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_patient_record(paciente_id: str, path: str = PRONTUARIOS_PATH) -> dict | None:
    prontuarios = load_prontuarios(path)
    return prontuarios.get(paciente_id)


def format_patient_summary(record: dict) -> str:
    if not record:
        return "Paciente não encontrado na base de prontuários."

    vitais = record.get("sinais_vitais", {})
    linhas = [
        f"Paciente {record['paciente_id']} ({record.get('idade')} anos, {record.get('sexo')}), "
        f"setor: {record.get('setor')}.",
        f"Sinais vitais: PA {vitais.get('pa_sistolica')}x{vitais.get('pa_diastolica')} mmHg, "
        f"FR {vitais.get('frequencia_respiratoria')} irpm, "
        f"Temp {vitais.get('temperatura_c')} C, Glasgow {vitais.get('glasgow')}.",
        f"Hipótese diagnóstica: {record.get('hipotese_diagnostica', 'não informada')}.",
        f"Exames pendentes: {', '.join(record.get('exames_pendentes', [])) or 'nenhum'}.",
        f"Comorbidades: {', '.join(record.get('comorbidades', [])) or 'nenhuma'}.",
        f"Alergias: {', '.join(record.get('alergias', [])) or 'nenhuma'}.",
    ]
    return "\n".join(linhas)
