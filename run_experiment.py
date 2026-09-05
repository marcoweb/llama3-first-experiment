#!/usr/bin/env python3
"""
run_experiment.py
Ponto de entrada unificado para execução do experimento de indexação automática:
Léxica (BM25) + Semântica (FAISS) + Reranqueamento LLaMA 3 (Ollama).
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from src.corpus import (
    extract_agriculture_data,
    load_corpus_dataset,
    get_controlled_vocabulary
)
from src.hybrid_index import HybridIndex, OllamaClient
from src.llm_selector import LLMSelector
from src.evaluator import (
    compute_document_metrics,
    compute_corpus_metrics,
    format_comparison_tables
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experimento de Indexação Automática com LLaMA 3 e Ollama (Artigo ENANCIB)"
    )
    parser.add_argument(
        "--corpus",
        choices=["C1", "C2", "C3"],
        default="C1",
        help="Subcorpus de teste para avaliação (padrão: C1)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Executa apenas em uma amostra de N documentos (ex: --sample 5)"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Apenas extrai dados e constrói/atualiza os índices BM25 e FAISS"
    )
    parser.add_argument(
        "--model",
        default="llama3:8b",
        help="Modelo LLM no Ollama (padrão: llama3:8b)"
    )
    parser.add_argument(
        "--embed-model",
        default="mxbai-embed-large",
        help="Modelo de Embeddings no Ollama para o FAISS (padrão: mxbai-embed-large)"
    )
    parser.add_argument(
        "--ollama-host",
        default="http://127.0.0.1:11434",
        help="URL do servidor Ollama (padrão: http://127.0.0.1:11434)"
    )
    parser.add_argument(
        "--vocab-source",
        default="all",
        choices=["all", "corpus", "train", "C1", "C2", "C3"],
        help="Fonte do vocabulário controlado: 'all' (~23k), 'corpus' (específico do corpus escolhido), 'train' (~21.4k) (padrão: all)"
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=35,
        help="Número total de candidatos híbridos enviados ao LLaMA 3 (padrão: 35)"
    )
    parser.add_argument(
        "--vocab-limit",
        type=int,
        default=None,
        help="Limita o tamanho do vocabulário indexado para testes rápidos (padrão: completo)"
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=20,
        help="Número máximo de descritores finais retornados pelo modelo (padrão: 20)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Diretório para salvar relatórios JSON de resultados (padrão: results)"
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Diretório de cache de índices e vetores (padrão: cache)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Diretório dos dados extraídos (padrão: data)"
    )
    parser.add_argument(
        "--zip-path",
        default=None,
        help="Caminho alternativo para o arquivo Corpora+Gold_Standard_Index.zip"
    )
    return parser.parse_args()


def find_or_extract_corpus(data_dir: Path, custom_zip: str = None) -> None:
    """Verifica se os dados já estão extraídos ou localiza o arquivo zip baixado."""
    gold_dir = data_dir / "gold_standard"
    if gold_dir.exists() and len(list(gold_dir.glob("*.txt"))) >= 4:
        return

    # Candidatos onde o arquivo zip pode estar
    possible_zips = [
        custom_zip,
        data_dir / "Corpora+Gold_Standard_Index.zip",
        data_dir / "corpus.zip",
        Path("/home/marco/.gemini/antigravity/brain/fe4ea8f2-d314-4cb2-9fc3-65c298252804/scratch/corpus.zip"),
    ]
    zip_to_use = None
    for z in possible_zips:
        if z and Path(z).exists():
            zip_to_use = Path(z)
            break

    if not zip_to_use:
        raise FileNotFoundError(
            "Arquivo compactado do Zenodo não encontrado. "
            "Baixe Corpora+Gold_Standard_Index.zip ou passe o caminho via --zip-path."
        )

    extract_agriculture_data(zip_to_use, data_dir)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" EXPERIMENTO DE INDEXAÇÃO AUTOMÁTICA (ENANCIB) - LLaMA 3 + OLLAMA")
    print("=" * 65)
    print(f"Subcorpus selecionado: {args.corpus}")
    print(f"Modelo de Geração:     {args.model}")
    print(f"Modelo de Embeddings:  {args.embed_model}")
    print(f"Servidor Ollama:       {args.ollama_host}")
    if args.sample:
        print(f"Modo Amostra:          {args.sample} documentos")
    print("=" * 65)

    # 1. Preparação dos dados
    print("\n[Etapa 1/4] Verificando dados e vocabulário controlado...")
    find_or_extract_corpus(data_dir, args.zip_path)

    vocab_source = args.vocab_source
    if vocab_source == "corpus":
        vocab_source = args.corpus

    vocabulary = get_controlled_vocabulary(data_dir, source=vocab_source)
    print(f"Vocabulário controlado carregado ({vocab_source}): {len(vocabulary)} termos únicos.")

    if args.vocab_limit:
        vocabulary = vocabulary[: args.vocab_limit]
        print(f"Vocabulário limitado a {len(vocabulary)} termos para teste.")

    # 2. Inicialização do Índice Híbrido (BM25 + FAISS)
    print("\n[Etapa 2/4] Carregando ou construindo Índice Híbrido (BM25 + FAISS)...")
    index_name = "vocab" if vocab_source == "all" else f"vocab_{vocab_source}"
    ollama_client = OllamaClient(host=args.ollama_host, embed_model=args.embed_model)
    hybrid_index = HybridIndex(
        vocabulary=vocabulary,
        cache_dir=cache_dir,
        name=index_name,
        ollama_client=ollama_client
    )
    hybrid_index.build_or_load_bm25()
    hybrid_index.build_or_load_faiss(max_terms=args.vocab_limit)

    if args.prepare_only:
        print("\nPreparação concluída com sucesso! (--prepare-only especificado)")
        sys.exit(0)

    # 3. Carregamento do dataset de teste
    print(f"\n[Etapa 3/4] Carregando documentos do subcorpus {args.corpus}...")
    dataset = load_corpus_dataset(args.corpus, data_dir)
    if args.sample is not None and args.sample > 0:
        dataset = dataset[: args.sample]
        print(f"Filtrado para os primeiros {len(dataset)} documentos (amostra de teste).")
    else:
        print(f"Total de documentos para indexação: {len(dataset)}")

    # 4. Processamento com LLaMA 3
    print(f"\n[Etapa 4/4] Executando pipeline híbrido + LLaMA 3...")
    selector = LLMSelector(model_name=args.model, ollama_host=args.ollama_host)

    results = []
    checkpoint_file = cache_dir / f"checkpoint_{args.corpus}.json"

    for gold_id, doc, gold_terms in tqdm(dataset, desc=f"Indexando {args.corpus}"):
        # Consulta para recuperação: Título + Resumo + Palavras-chave
        query_text = doc.full_query_text

        # Recuperação Híbrida: BM25 + FAISS com fusão RRF
        candidates = hybrid_index.retrieve_hybrid_candidates(
            query=query_text,
            top_k_lex=30,
            top_k_sem=30,
            total_candidates=args.candidates
        )

        # Seleção e Reranqueamento com LLaMA 3
        predicted_terms = selector.select_descriptors(
            title=doc.title,
            abstract=doc.abstract,
            candidates=candidates,
            max_output_terms=args.max_terms
        )

        # Avaliação do documento
        doc_metrics = compute_document_metrics(predicted_terms, gold_terms)

        doc_result = {
            "gold_id": gold_id,
            "doc_id": doc.doc_id,
            "title": doc.title,
            "candidates": candidates,
            "predicted": predicted_terms,
            "gold": gold_terms,
            "metrics": doc_metrics
        }
        results.append(doc_result)

    # Cálculo das métricas consolidadas do corpus
    corpus_metrics = compute_corpus_metrics(results)

    # Exibição das Tabelas 1 e 2 formatadas
    summary_text = format_comparison_tables(args.corpus, corpus_metrics)
    print("\n" + summary_text)

    # Salvar resultados em JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"results_{args.corpus}_{timestamp}.json"
    out_path = output_dir / out_filename

    final_payload = {
        "corpus": args.corpus,
        "sample_size": len(results),
        "timestamp": timestamp,
        "model": args.model,
        "embed_model": args.embed_model,
        "metrics": corpus_metrics,
        "detailed_results": results
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)

    print(f"\nRelatório detalhado salvo em: {out_path}")


if __name__ == "__main__":
    main()
