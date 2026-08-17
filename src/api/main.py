"""API FastAPI opcional para expor o assistente médico fora do notebook.

Uso:
    uvicorn src.api.main:app --reload

O modelo fine-tuned é carregado uma única vez na inicialização (lifespan).
Se o adaptador LoRA ainda não existir (fine-tuning não executado), a API cai
automaticamente para o fallback extrativo baseado apenas no RAG.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from ..agent.graph import run_assistente
from ..agent.llm_client import make_hf_pipeline_generator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTER_DIR = os.path.join(BASE_DIR, "models", "llm-medico-lora")
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_state = {"generator": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.isdir(ADAPTER_DIR):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        pipe = pipeline("text-generation", model=finetuned_model, tokenizer=tokenizer)
        _state["generator"] = make_hf_pipeline_generator(pipe)
        print("Modelo fine-tuned carregado.")
    else:
        print("Adaptador LoRA não encontrado — usando fallback extrativo (sem LLM).")
    yield
    _state.clear()


app = FastAPI(title="Assistente Médico — Hospital Vida Plena", lifespan=lifespan)


class ConsultaRequest(BaseModel):
    pergunta: str
    paciente_id: str | None = None


@app.get("/")
async def root():
    return {"message": "Assistente Médico — Hospital Vida Plena (Tech Challenge Fase 3)"}


@app.post("/consulta")
async def consulta(req: ConsultaRequest):
    resultado = run_assistente(
        pergunta=req.pergunta,
        paciente_id=req.paciente_id,
        generator=_state["generator"],
    )
    return {
        "resposta": resultado["resposta_final"],
        "alertas": resultado["alertas"],
        "fontes": [c["source"] for c in resultado.get("contexto_protocolo", [])],
        "guardrail_flags": resultado["guardrail_flags"],
        "bloqueado": resultado["bloqueado"],
    }
