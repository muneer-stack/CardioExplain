from __future__ import annotations
import hashlib
import json
import re
import shutil
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    DATA_PATH,
    INDEX_DIR,
    TERM_CATALOG_PATH,
    INDEX_MANIFEST_PATH,
    EMBEDDING_MODEL,
)


def normalize_text(value: object) -> str:
    """
    Vereinheitlicht Textfelder, ohne ihren Inhalt fachlich zu verändern.
    """
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_term(value: object) -> str:
    """
    Normalisierte Schreibweise für spätere exakte Vergleiche.
    Die sichtbare Originalschreibweise bleibt separat erhalten.
    """
    return normalize_text(value).lower()


def calculate_file_hash(path: Path) -> str:
    """
    Berechnet einen SHA-256-Hash der Datenbank.
    Dadurch kann später nachvollzogen werden,
    mit welcher konkreten Datei der Index erstellt wurde.
    """
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def load_and_prepare_database(path: Path) -> pd.DataFrame:
    """
    Lädt die finale bilinguale README-Datenbank und
    prüft die für das RAG-System benötigten Spalten.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Die Datenbank wurde nicht gefunden:\n{path}"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
    )

    required_columns = {
        "record_id",
        "mention_count",
        "ann_text_en",
        "ann_text_de",
        "general_definition_en",
        "lay_definition_en",
        "lay_definition_de",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Folgende benötigte Spalten fehlen in der Datenbank: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()

    text_columns = [
        "record_id",
        "ann_text_en",
        "ann_text_de",
        "general_definition_en",
        "lay_definition_en",
        "lay_definition_de",
    ]

    for column in text_columns:
        df[column] = df[column].map(normalize_text)


    df["mention_count"] = pd.to_numeric(
        df["mention_count"],
        errors="coerce",
    ).fillna(0).astype(int)


    df["term_norm_en"] = df["ann_text_en"].map(normalize_term)
    df["term_norm_de"] = df["ann_text_de"].map(normalize_term)


    df = df[
        (df["record_id"] != "")
        & ((df["ann_text_en"] != "") | (df["ann_text_de"] != ""))
    ].copy()

   
    duplicate_ids = df[df["record_id"].duplicated(keep=False)]

    if not duplicate_ids.empty:
        examples = ", ".join(
            duplicate_ids["record_id"].head(10).tolist()
        )

        raise ValueError(
            "record_id ist nicht eindeutig. "
            f"Beispiele: {examples}"
        )

    return df


def build_documents(
    df: pd.DataFrame,
) -> tuple[list[Document], pd.DataFrame]:
    """
    Erstellt genau ein FAISS-Dokument pro record_id.

    Es findet bewusst KEINE Gruppierung nach Fachbegriff statt.
    Dadurch bleiben unterschiedliche README-Einträge,
    Definitionen und Bedeutungen getrennt erhalten.
    """
    documents: list[Document] = []
    catalog_rows: list[dict] = []

    for _, row in df.iterrows():

        record_id = row["record_id"]

        term_en = row["ann_text_en"]
        term_de = row["ann_text_de"]

        general_definition_en = row["general_definition_en"]
        lay_definition_en = row["lay_definition_en"]
        lay_definition_de = row["lay_definition_de"]

        mention_count = int(row["mention_count"])

        term_norm_en = row["term_norm_en"]
        term_norm_de = row["term_norm_de"]

        content = f"""
Fachbegriff Deutsch:
{term_de if term_de else "Nicht vorhanden"}

Fachbegriff Englisch:
{term_en if term_en else "Nicht vorhanden"}

Allgemeine medizinische Definition aus README:
{general_definition_en if general_definition_en else "Nicht vorhanden"}

Laienverständliche Erklärung Deutsch:
{lay_definition_de if lay_definition_de else "Nicht vorhanden"}

Laienverständliche Erklärung Englisch:
{lay_definition_en if lay_definition_en else "Nicht vorhanden"}
""".strip()

        metadata = {
            "record_id": record_id,
            "term_en": term_en,
            "term_de": term_de,
            "term_norm_en": term_norm_en,
            "term_norm_de": term_norm_de,
            "mention_count": mention_count,
        }

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

        catalog_rows.append(metadata)

    catalog = pd.DataFrame(catalog_rows)

    return documents, catalog


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def save_manifest(
    database_rows: int,
    database_hash: str,
) -> None:
    """
    Speichert Informationen über den Index.
    Das hilft später bei Reproduzierbarkeit und Evaluation.
    """
    manifest = {
        "database_file": DATA_PATH.name,
        "database_rows": database_rows,
        "database_sha256": database_hash,
        "embedding_model": EMBEDDING_MODEL,
        "index_unit": "one_document_per_record_id",
    }

    INDEX_MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        INDEX_MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:

    print("=" * 70)
    print("CardioExplain – Aufbau des README-FAISS-Index")
    print("=" * 70)

    print()
    print("Verwendete Datenbank:")
    print(DATA_PATH)

    print()
    print("Lade Datenbank ...")

    df = load_and_prepare_database(DATA_PATH)

    print(f"Geladene Datensätze: {len(df):,}")

    print(
        "Unterschiedliche englische Fachbegriffe: "
        f"{df['term_norm_en'].nunique():,}"
    )

    print(
        "Unterschiedliche deutsche Fachbegriffe: "
        f"{df['term_norm_de'].nunique():,}"
    )

    print(
        "Summe mention_count: "
        f"{df['mention_count'].sum():,}"
    )

    print()
    print("Erzeuge ein Retrieval-Dokument pro record_id ...")

    documents, catalog = build_documents(df)

    print(f"Index-Dokumente: {len(documents):,}")

    if len(documents) != len(df):
        raise RuntimeError(
            "Anzahl der Index-Dokumente stimmt nicht "
            "mit der Anzahl der Datensätze überein."
        )

    print()
    print(f"Lade Embedding-Modell:")
    print(EMBEDDING_MODEL)

    embeddings = get_embeddings()

    print()
    print("Erstelle neuen FAISS-Index ...")

    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = FAISS.from_documents(
        documents,
        embeddings,
    )

    vector_store.save_local(
        str(INDEX_DIR)
    )

    print("FAISS-Index gespeichert.")

    print()
    print("Speichere Term-Katalog ...")

    TERM_CATALOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog.to_csv(
        TERM_CATALOG_PATH,
        index=False,
        encoding="utf-8",
    )

    database_hash = calculate_file_hash(
        DATA_PATH
    )

    save_manifest(
        database_rows=len(df),
        database_hash=database_hash,
    )

    print()
    print("=" * 70)
    print("Index erfolgreich erstellt")
    print("=" * 70)

    print(f"Datenbank:          {DATA_PATH.name}")
    print(f"Datensätze:         {len(df):,}")
    print(f"Index-Dokumente:    {len(documents):,}")
    print(f"FAISS-Index:        {INDEX_DIR}")
    print(f"Term-Katalog:       {TERM_CATALOG_PATH}")
    print(f"Index-Manifest:     {INDEX_MANIFEST_PATH}")


if __name__ == "__main__":
    main()