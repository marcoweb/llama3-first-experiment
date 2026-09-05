"""
src/corpus.py
Módulo para extração, parsing e carregamento do corpus de agricultura e padrões de referência (Gold Standards).
"""

import os
import re
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

TAG_PATTERNS = {
    "title": re.compile(r"#ITI#(.*?)#FTI#", re.DOTALL),
    "abstract": re.compile(r"#IRE#(.*?)#FRE#", re.DOTALL),
    "keywords": re.compile(r"#IPC#(.*?)#FPC#", re.DOTALL),
    "references": re.compile(r"#IRF#(.*?)#FRF#", re.DOTALL),
}


class Document:
    def __init__(self, doc_id: str, title: str, abstract: str, keywords: List[str], references: str = ""):
        self.doc_id = doc_id
        self.title = title.strip()
        self.abstract = abstract.strip()
        self.keywords = [k.strip() for k in keywords if k.strip()]
        self.references = references.strip()

    @property
    def full_query_text(self) -> str:
        """Texto consolidado para busca lexical e semântica."""
        parts = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        return "\n".join(parts)

    def __repr__(self):
        return f"<Document {self.doc_id}: {self.title[:40]}...>"


def parse_document_text(content: str, doc_id: str = "") -> Document:
    """Extrai campos das tags estruturadas do arquivo de texto."""
    title_m = TAG_PATTERNS["title"].search(content)
    abstract_m = TAG_PATTERNS["abstract"].search(content)
    keywords_m = TAG_PATTERNS["keywords"].search(content)
    ref_m = TAG_PATTERNS["references"].search(content)

    title = title_m.group(1).strip() if title_m else ""
    abstract = abstract_m.group(1).strip() if abstract_m else ""
    keywords_raw = keywords_m.group(1).strip() if keywords_m else ""
    references = ref_m.group(1).strip() if ref_m else ""

    # Separar palavras-chave por ponto-e-vírgula ou vírgula
    if ";" in keywords_raw:
        keywords = [k.strip() for k in keywords_raw.split(";") if k.strip()]
    elif "," in keywords_raw:
        keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
    else:
        keywords = [keywords_raw] if keywords_raw else []

    return Document(
        doc_id=doc_id,
        title=title,
        abstract=abstract,
        keywords=keywords,
        references=references
    )


def load_gold_standard_file(file_path: Path) -> Dict[str, List[str]]:
    """
    Carrega o arquivo Gold Standard (ex: Gold_Standard_Index_1_NAL_Agricola_500_documents.txt).
    Retorna mapeamento: 'D1' -> ['descritor 1', 'descritor 2', ...]
    """
    gold_map = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) == 2:
                doc_key = parts[0].strip()
                terms = [t.strip() for t in parts[1].split(",") if t.strip()]
                gold_map[doc_key] = terms
    return gold_map


def extract_agriculture_data(zip_source_path: Path, data_dir: Path) -> None:
    """
    Extrai do zip principal do Zenodo (Corpora+Gold_Standard_Index.zip) os subconjuntos de Agricultura.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    corpora_dir = data_dir / "corpora"
    gold_dir = data_dir / "gold_standard"
    corpora_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lendo pacote principal: {zip_source_path}")
    with zipfile.ZipFile(zip_source_path, "r") as main_zip:
        # 1. Extrair Gold Standards
        for name in main_zip.namelist():
            if "Gold Standard Index (Agricultural domain)" in name and name.endswith(".txt"):
                dest_file = gold_dir / Path(name).name
                if not dest_file.exists():
                    print(f"  -> Extraindo Gold Standard: {dest_file.name}")
                    with open(dest_file, "wb") as f:
                        f.write(main_zip.read(name))

        # 2. Extrair subcorpora C1, C2, C3
        subcorpora = {
            "C1": "Corpora+Gold_Standard_Index/Agriculture/Agricultural Corpus/Agricultural Corpus_1_500_documents_TI_AB_KW_RE.zip",
            "C2": "Corpora+Gold_Standard_Index/Agriculture/Agricultural Corpus/Agricultural Corpus_2_500_documents_TI_AB_KW_RE.zip",
            "C3": "Corpora+Gold_Standard_Index/Agriculture/Agricultural Corpus/Agricultural Corpus_3_500_documents_TI_AB_KW_RE.zip",
        }

        for sub_name, zip_internal_path in subcorpora.items():
            sub_target_dir = corpora_dir / sub_name
            if sub_target_dir.exists() and len(list(sub_target_dir.glob("*.txt"))) == 500:
                print(f"  -> Subcorpus {sub_name} já extraído ({len(list(sub_target_dir.glob('*.txt')))} arquivos).")
                continue

            sub_target_dir.mkdir(parents=True, exist_ok=True)
            print(f"  -> Descompactando Subcorpus {sub_name}...")
            nested_zip_data = main_zip.read(zip_internal_path)
            with zipfile.ZipFile(io.BytesIO(nested_zip_data), "r") as nested_zip:
                for item in nested_zip.namelist():
                    if item.endswith(".txt"):
                        file_bytes = nested_zip.read(item)
                        filename = Path(item).name
                        with open(sub_target_dir / filename, "wb") as out_f:
                            out_f.write(file_bytes)
            print(f"     Extraídos {len(list(sub_target_dir.glob('*.txt')))} documentos em {sub_target_dir}")


def load_corpus_dataset(corpus_name: str, data_dir: Path) -> List[Tuple[str, Document, List[str]]]:
    """
    Carrega o dataset especificado ('C1', 'C2' ou 'C3').
    Retorna lista ordenada de tuplas: (gold_id 'D1'..'D500', Document, list_of_gold_descriptors).
    """
    corpora_dir = data_dir / "corpora" / corpus_name
    gold_dir = data_dir / "gold_standard"

    gold_filename_map = {
        "C1": "Gold_Standard_Index_1_NAL_Agricola_500_documents.txt",
        "C2": "Gold_Standard_Index_2_NAL_Agricola_500_documents.txt",
        "C3": "Gold_Standard_Index_3_NAL_Agricola_500_documents.txt",
    }
    gold_path = gold_dir / gold_filename_map[corpus_name]
    if not gold_path.exists():
        raise FileNotFoundError(f"Arquivo Gold Standard não encontrado: {gold_path}")

    gold_map = load_gold_standard_file(gold_path)

    # Ordenar arquivos txt numericamente (ex: D8001.txt ... D8500.txt)
    txt_files = sorted(
        corpora_dir.glob("*.txt"),
        key=lambda p: int(re.search(r"\d+", p.stem).group(0)) if re.search(r"\d+", p.stem) else 0
    )

    dataset = []
    for idx, fpath in enumerate(txt_files, start=1):
        gold_id = f"D{idx}"
        gold_terms = gold_map.get(gold_id, [])
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        doc = parse_document_text(content, doc_id=fpath.stem)
        dataset.append((gold_id, doc, gold_terms))

    return dataset


def get_controlled_vocabulary(data_dir: Path, source: str = "all") -> List[str]:
    """
    Compila o vocabulário controlado a partir dos Gold Standards do AGRICOLA disponíveis no corpus.
    Parâmetro source:
      - 'all': todos os termos únicos de todos os gold standards (~23k termos).
      - 'train': termos do conjunto de treino (8000 documentos, ~21.4k termos).
      - 'C1', 'C2', 'C3': termos específicos do subcorpus de teste correspondente.
    """
    gold_dir = data_dir / "gold_standard"
    all_terms: Set[str] = set()

    file_mapping = {
        "C1": ["Gold_Standard_Index_1_NAL_Agricola_500_documents.txt"],
        "C2": ["Gold_Standard_Index_2_NAL_Agricola_500_documents.txt"],
        "C3": ["Gold_Standard_Index_3_NAL_Agricola_500_documents.txt"],
        "TRAIN": ["Gold_Standard_Index_NAL_Agricola_8000_documents.txt"],
    }

    if source.upper() in file_mapping:
        target_files = [gold_dir / f for f in file_mapping[source.upper()]]
    else:
        target_files = list(gold_dir.glob("*.txt"))

    for gfile in target_files:
        if gfile.exists():
            gmap = load_gold_standard_file(gfile)
            for terms in gmap.values():
                for t in terms:
                    cleaned = t.strip()
                    if cleaned:
                        all_terms.add(cleaned)

    sorted_vocab = sorted(list(all_terms))
    return sorted_vocab

