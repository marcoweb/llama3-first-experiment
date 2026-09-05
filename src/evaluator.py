"""
src/evaluator.py
Módulo de avaliação quantitativa: Precisão, Revocação, F-Measure e reprodução das Tabelas 1 e 2 do artigo.
"""

from typing import List, Dict, Set, Any, Tuple
import numpy as np


def normalize_term(term: str) -> str:
    """Normaliza o descritor para comparação documental consistente."""
    return term.strip().lower()


def compute_document_metrics(predicted: List[str], gold: List[str]) -> Dict[str, float]:
    """Calcula Precisão, Revocação e Medida-F para um único documento."""
    pred_set = {normalize_term(t) for t in predicted if t.strip()}
    gold_set = {normalize_term(t) for t in gold if t.strip()}

    if not pred_set or not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f_measure": 0.0, "hits": 0, "pred_count": len(pred_set), "gold_count": len(gold_set)}

    hits = len(pred_set.intersection(gold_set))
    precision = (hits / len(pred_set)) * 100.0
    recall = (hits / len(gold_set)) * 100.0
    if precision + recall > 0:
        f_measure = (2.0 * precision * recall) / (precision + recall)
    else:
        f_measure = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
        "hits": hits,
        "pred_count": len(pred_set),
        "gold_count": len(gold_set)
    }


def compute_corpus_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula médias macro do corpus e distribuição em faixas da Tabela 2 do artigo."""
    precisions = [r["metrics"]["precision"] for r in results]
    recalls = [r["metrics"]["recall"] for r in results]
    f_measures = [r["metrics"]["f_measure"] for r in results]

    avg_precision = float(np.mean(precisions)) if precisions else 0.0
    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    avg_f_measure = float(np.mean(f_measures)) if f_measures else 0.0

    # Contagem de documentos nas 6 faixas percentuais da Tabela 2
    # Faixas: 0%, 1-15%, 16-30%, 31-49%, 50-74%, >=75%
    distribution = {
        "0%": 0,
        "1-15%": 0,
        "16-30%": 0,
        "31-49%": 0,
        "50-74%": 0,
        ">=75%": 0
    }

    for f in f_measures:
        if f == 0.0:
            distribution["0%"] += 1
        elif 0.0 < f <= 15.0:
            distribution["1-15%"] += 1
        elif 15.0 < f <= 30.0:
            distribution["16-30%"] += 1
        elif 30.0 < f <= 49.999:
            distribution["31-49%"] += 1
        elif 50.0 <= f < 75.0:
            distribution["50-74%"] += 1
        else:
            distribution[">=75%"] += 1

    return {
        "total_documents": len(results),
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f_measure": avg_f_measure,
        "distribution": distribution
    }


# Valores de referência reportados no Artigo (ENANCIB)
PAPER_BENCHMARKS = {
    "Table_1": {
        "C1": {
            "SISA": {"precision": 74.26, "recall": 40.38, "f_measure": 52.83},
            "Annif_MLLM": {"precision": 74.26, "recall": 40.38, "f_measure": 50.90},
            "Annif_Ensemble": {"precision": 64.30, "recall": 34.66, "f_measure": 43.80},
            "LLM_Hybrid_Paper": {"precision": 33.28, "recall": 17.82, "f_measure": 22.50}
        },
        "C2": {
            "SISA": {"precision": 80.15, "recall": 50.36, "f_measure": 64.05},
            "Annif_MLLM": {"precision": 80.15, "recall": 50.36, "f_measure": 60.20},
            "Annif_Ensemble": {"precision": 66.87, "recall": 42.22, "f_measure": 50.40},
            "LLM_Hybrid_Paper": {"precision": 32.72, "recall": 20.32, "f_measure": 24.54}
        },
        "C3": {
            "SISA": {"precision": 81.45, "recall": 51.05, "f_measure": 65.31},
            "Annif_MLLM": {"precision": 59.26, "recall": 69.06, "f_measure": 62.84},
            "Annif_Ensemble": {"precision": 68.88, "recall": 43.03, "f_measure": 51.80},
            "LLM_Hybrid_Paper": {"precision": 34.73, "recall": 21.64, "f_measure": 26.11}
        }
    },
    "Table_2": {
        "C1": {"0%": 12, "1-15%": 129, "16-30%": 256, "31-49%": 91, "50-74%": 12, ">=75%": 0},
        "C2": {"0%": 14, "1-15%": 117, "16-30%": 221, "31-49%": 126, "50-74%": 22, ">=75%": 0},
        "C3": {"0%": 17, "1-15%": 100, "16-30%": 208, "31-49%": 148, "50-74%": 27, ">=75%": 0}
    }
}


def format_comparison_tables(corpus_name: str, exp_metrics: Dict[str, Any]) -> str:
    """Gera tabelas comparativas formatadas em texto/markdown."""
    out = []
    out.append("=" * 78)
    out.append(f" RESULTADOS DO EXPERIMENTO vs. ARTIGO ENANCIB (Subcorpus: {corpus_name})")
    out.append("=" * 78)

    # Tabela 1
    ref = PAPER_BENCHMARKS["Table_1"].get(corpus_name, {}).get("LLM_Hybrid_Paper", {})
    out.append("\n[TABELA 1] Comparação de Desempenho Global:")
    out.append(f"{'Sistema / Fonte':<25} | {'Precisão (%)':<14} | {'Revocação (%)':<14} | {'Medida-F (%)':<14}")
    out.append("-" * 75)
    out.append(
        f"{'LLM Híbrido (Executado)':<25} | {exp_metrics['avg_precision']:<14.2f} | "
        f"{exp_metrics['avg_recall']:<14.2f} | {exp_metrics['avg_f_measure']:<14.2f}"
    )
    if ref:
        out.append(
            f"{'LLM Híbrido (Artigo)':<25} | {ref['precision']:<14.2f} | "
            f"{ref['recall']:<14.2f} | {ref['f_measure']:<14.2f}"
        )
    out.append("-" * 75)

    # Tabela 2
    dist = exp_metrics["distribution"]
    ref_dist = PAPER_BENCHMARKS["Table_2"].get(corpus_name, {})
    total_docs = exp_metrics["total_documents"]
    out.append(f"\n[TABELA 2] Distribuição dos Documentos por Faixas de Medida-F (N={total_docs}):")
    out.append(f"{'Origem':<18} | {'f-m 0%':<8} | {'1-15%':<8} | {'16-30%':<8} | {'31-49%':<8} | {'50-74%':<8} | {'>=75%':<8}")
    out.append("-" * 75)
    out.append(
        f"{'Executado':<18} | {dist['0%']:<8} | {dist['1-15%']:<8} | {dist['16-30%']:<8} | "
        f"{dist['31-49%']:<8} | {dist['50-74%']:<8} | {dist['>=75%']:<8}"
    )
    if ref_dist:
        out.append(
            f"{'Artigo Llama 3':<18} | {ref_dist['0%']:<8} | {ref_dist['1-15%']:<8} | {ref_dist['16-30%']:<8} | "
            f"{ref_dist['31-49%']:<8} | {ref_dist['50-74%']:<8} | {ref_dist['>=75%']:<8}"
        )
    out.append("=" * 78)
    return "\n".join(out)

