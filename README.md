# Experimento de Indexação Automática com LLaMA 3 e Ollama

Este repositório contém a implementação minimalista e reproduzível da arquitetura baseada em Modelos de Linguagem de Grande Porte (LLM).

---

## 1. Visão Geral da Arquitetura

O pipeline implementa a abordagem **LLM em Pipeline Híbrido**:

```mermaid
flowchart LR
    Doc[Documento: Título + Resumo + Palavras-Chave] --> Hybrid[Recuperação Híbrida]
    
    subgraph Indexação do Vocabulário [NAL Thesaurus / AGRICOLA]
        BM25[Índice Léxico: BM25]
        FAISS[Índice Semântico: FAISS + Embeddings]
    end
    
    Indexação do Vocabulário --> Hybrid
    Hybrid -->|Fusão RRF| Cand[Descritores Candidatos]
    Cand --> LLaMA[Reranqueamento LLaMA 3 via Ollama]
    Doc --> LLaMA
    LLaMA --> Pred[Descritores Finais Autorizados]
    Pred --> Eval[Avaliação vs. Gold Standard: Precisão, Revocação e Medida-F]
```

1. **Vocabulário Controlado**: Baseado nos descritores do NAL Thesaurus derivados da base AGRICOLA (~23.000 termos únicos).
2. **Recuperação Léxica**: Algoritmo Best Matching 25 (`rank-bm25`) com filtragem de stopwords.
3. **Recuperação Semântica**: Vetores densos gerados via Ollama (`mxbai-embed-large`) e indexados em alta velocidade com `faiss-cpu`.
4. **Agregação e Fusão**: Combinação dos melhores termos via *Reciprocal Rank Fusion* (RRF).
5. **Reranqueamento e Seleção com LLaMA 3**: O modelo `llama3:8b` recebe o documento e a lista estrita de candidatos, filtrando e ordenando os descritores mais representativos.
6. **Avaliação**: Cálculo de Precisão, Revocação, Medida-F.

---

## 2. Pré-requisitos

- Python 3.10+ (testado em Python 3.14)
- [Ollama](https://ollama.com/) instalado e em execução:
  ```bash
  ollama serve
  ```
- Modelos no Ollama:
  ```bash
  ollama pull llama3:8b
  ollama pull mxbai-embed-large
  ```

---

## 3. Instalação Rápida

Execute o script de configuração para criar o ambiente virtual e instalar as dependências:

```bash
./setup_env.sh
source .venv/bin/activate
```

Ou instale manualmente via pip:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Como Executar

### 4.1. Teste Rápido em Amostra (Recomendado para início)
Para verificar o pipeline em poucos documentos (ex: 5 documentos do subcorpus C1):

```bash
python run_experiment.py --corpus C1 --sample 5
```

### 4.2. Preparar e Cachear o Índice do Vocabulário Completo
Gera e armazena os embeddings e o índice FAISS em `cache/` para que as execuções seguintes carreguem instantaneamente:

```bash
python run_experiment.py --prepare-only --vocab-source all
```

### 4.3. Execução Completa nos Subcorpora C1, C2 ou C3 (500 documentos cada)
```bash
# Executar subcorpus C1
python run_experiment.py --corpus C1

# Executar subcorpus C2
python run_experiment.py --corpus C2

# Executar subcorpus C3
python run_experiment.py --corpus C3
```

---

## 5. Parâmetros da Linha de Comando (CLI)

| Argumento | Padrão | Descrição |
|---|---|---|
| `--corpus` | `C1` | Subcorpus de teste (`C1`, `C2` ou `C3`) |
| `--sample N` | `None` | Processa apenas os primeiros $N$ documentos |
| `--vocab-source` | `all` | Fonte de termos: `all` (~23k termos), `corpus` (específico do corpus) ou `train` |
| `--candidates` | `35` | Quantidade de candidatos híbridos enviados ao LLaMA 3 |
| `--max-terms` | `20` | Quantidade máxima de descritores finais retornados |
| `--model` | `llama3:8b` | Nome do modelo no Ollama |
| `--embed-model` | `mxbai-embed-large` | Modelo de embeddings no Ollama |
| `--prepare-only` | `False` | Apenas gera/atualiza cache de índices e sai |
| `--output-dir` | `results` | Diretório onde os relatórios JSON são gravados |

---

