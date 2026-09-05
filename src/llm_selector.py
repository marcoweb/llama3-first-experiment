"""
src/llm_selector.py
Módulo de seleção e reranqueamento de descritores com LLaMA 3 via Ollama.
Garante estrita aderência ao vocabulário controlado filtrando apenas os candidatos aprovados.
"""

import json
import re
from typing import List, Dict, Any, Optional
import requests


SYSTEM_PROMPT = """You are an expert subject indexer and librarian specialized in Agricultural Science and the NAL (National Agricultural Library) Controlled Thesaurus.
Your task is to assign the most relevant, accurate subject descriptors to a scientific document based on its title and abstract.

CRITICAL CONSTRAINTS:
1. You MUST ONLY select descriptors from the provided list of Candidate Descriptors.
2. Do NOT invent, modify, or extrapolate descriptors outside the candidate list.
3. Select the most salient terms that best represent the core subjects of the article (typically between 8 to 20 descriptors).
4. Respond ONLY with a valid JSON array of strings containing your selected descriptors, sorted by relevance. Example format:
["descriptor 1", "descriptor 2", "descriptor 3"]
"""


class LLMSelector:
    def __init__(
        self,
        model_name: str = "llama3:8b",
        ollama_host: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        timeout: int = 120
    ):
        self.model_name = model_name
        self.ollama_host = ollama_host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def select_descriptors(
        self,
        title: str,
        abstract: str,
        candidates: List[str],
        max_output_terms: int = 20
    ) -> List[str]:
        """
        Submete o documento e os candidatos ao LLaMA 3 para seleção temática e reranqueamento.
        """
        if not candidates:
            return []

        # Mapa case-insensitive para normalização e validação estrita
        candidate_map = {c.strip().lower(): c.strip() for c in candidates}
        candidate_list_text = "\n".join([f"- {c}" for c in candidates])

        user_content = f"""Document Title:
{title}

Document Abstract:
{abstract}

Candidate Descriptors (from NAL Controlled Vocabulary):
{candidate_list_text}

Task:
Select the most accurate and relevant descriptors from the Candidate Descriptors list above. Return ONLY a JSON array of selected strings."""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "options": {
                "temperature": self.temperature,
            },
            "stream": False,
            "format": "json"
        }

        try:
            resp = requests.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            if resp.status_code != 200:
                print(f"[Aviso] Ollama retornou código {resp.status_code}: {resp.text}")
                return candidates[:max_output_terms]

            data = resp.json()
            raw_reply = data.get("message", {}).get("content", "").strip()

            # Parser de JSON
            selected_terms = []
            try:
                parsed = json.loads(raw_reply)
                if isinstance(parsed, list):
                    selected_terms = parsed
                elif isinstance(parsed, dict):
                    # Às vezes modelos retornam {"descriptors": [...]}
                    for val in parsed.values():
                        if isinstance(val, list):
                            selected_terms = val
                            break
            except json.JSONDecodeError:
                # Tentar extrair lista por regex caso o JSON esteja incompleto
                matches = re.findall(r'"([^"]+)"', raw_reply)
                if matches:
                    selected_terms = matches

            # Filtro e validação estrita: o termo DEVE pertencer à lista de candidatos
            validated = []
            for term in selected_terms:
                term_str = str(term).strip()
                term_lower = term_str.lower()
                if term_lower in candidate_map:
                    matched_orig = candidate_map[term_lower]
                    if matched_orig not in validated:
                        validated.append(matched_orig)

            # Se a LLM selecionou muito poucos ou nenhum termo válido, complementa com o topo dos candidatos
            if len(validated) < 3:
                for cand in candidates:
                    if cand not in validated:
                        validated.append(cand)
                    if len(validated) >= max(len(candidates) // 2, 5):
                        break

            return validated[:max_output_terms]

        except Exception as e:
            print(f"[Aviso] Erro durante chamada ao LLaMA 3: {e}")
            # Fallback seguro: retorna os primeiros candidatos do índice híbrido
            return candidates[:max_output_terms]

