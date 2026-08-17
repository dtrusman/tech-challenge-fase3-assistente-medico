"""Regras simples de alerta clínico, derivadas dos protocolos internos.

Implementam de forma programática os limiares de "ALERTAS OBRIGATÓRIOS AO
TIME" descritos nos protocolos (data/raw/protocolos_hospitalares), permitindo
que o grafo LangGraph acione alertas automaticamente a partir dos sinais
vitais e exames do prontuário do paciente — sem depender do LLM para decidir
esses limiares numéricos.
"""


def avaliar_alertas(prontuario: dict | None) -> list[str]:
    if not prontuario:
        return []

    alertas = []
    vitais = prontuario.get("sinais_vitais", {})
    resultados = prontuario.get("exames_resultados", {})
    hipotese = (prontuario.get("hipotese_diagnostica") or "").lower()

    # qSOFA (PROT-INF-001): FR >= 22, PAS <= 100, Glasgow < 15
    qsofa = 0
    if (vitais.get("frequencia_respiratoria") or 0) >= 22:
        qsofa += 1
    if (vitais.get("pa_sistolica") or 999) <= 100:
        qsofa += 1
    if (vitais.get("glasgow") or 15) < 15:
        qsofa += 1
    if qsofa >= 2:
        alertas.append(
            f"ALERTA CRÍTICO (qSOFA = {qsofa}): critérios de sepse presentes — "
            "acionar PROT-INF-001 (pacote da primeira hora)."
        )

    # Crise hipertensiva (PROT-CARD-004): PAS >= 180 ou PAD >= 120
    if (vitais.get("pa_sistolica") or 0) >= 180 or (vitais.get("pa_diastolica") or 0) >= 120:
        alertas.append(
            "ALERTA: pressão arterial em faixa de urgência/emergência hipertensiva "
            "(>= 180x120 mmHg) — acionar PROT-CARD-004."
        )

    # Dengue com sinal de alarme (PROT-INF-007)
    if "dengue" in hipotese and resultados.get("prova_do_laco") == "positiva":
        alertas.append(
            "ALERTA: suspeita de dengue com prova do laço positiva — reclassificar "
            "risco e observar sinais de alarme, conforme PROT-INF-007."
        )

    # Lactato crítico (PROT-INF-001)
    lactato = resultados.get("lactato")
    if isinstance(lactato, (int, float)) and lactato >= 4:
        alertas.append(
            f"ALERTA CRÍTICO: lactato sérico = {lactato} mmol/L (>= 4) — risco de "
            "choque séptico, conforme PROT-INF-001."
        )

    # Glicemia crítica (PROT-END-002)
    glicemia = resultados.get("glicemia")
    if isinstance(glicemia, (int, float)) and glicemia < 70:
        alertas.append(
            f"ALERTA CRÍTICO: glicemia = {glicemia} mg/dL (< 70) — risco de "
            "hipoglicemia, conforme PROT-END-002."
        )

    return alertas
