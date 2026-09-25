"""
RAG-Logik für CardioExplain.

Das System nutzt die bilinguale README-Wissensbasis,
den lokalen Kontext eines deutschsprachigen Arztbriefs
und semantisches Retrieval zur Erklärung
kardiologischer Fachbegriffe.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from config import (
    DATA_PATH,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    INDEX_DIR,
    INDEX_MANIFEST_PATH,
    MAX_PATIENT_TEXT_CHARS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


SYSTEM_PROMPT = """
Du bist CardioExplain, ein KI-gestütztes Assistenzsystem.

Deine Aufgabe ist es, kardiologische Fachbegriffe aus deutschsprachigen
Arztbriefen für medizinische Laien verständlich zu erklären.

Regeln:
- Antworte immer auf Deutsch.
- Formuliere klar, kurz und verständlich.
- Orientiere dich ungefähr an einem sprachlichen Niveau der 7. bis 8. Klasse.
- Nutze für medizinische Aussagen ausschließlich die bereitgestellten
  README-Informationen und den bereitgestellten Arztbriefauszug.
- Ergänze keine medizinischen Informationen aus eigenem Wissen,
  wenn sie nicht durch diese Informationen gestützt werden.
- Gib keine Diagnose, Therapieentscheidung oder individuelle
  medizinische Empfehlung.
- Wenn eine Abkürzung oder ein Begriff mehrere Bedeutungen haben kann,
  nutze den Arztbriefkontext zur Einordnung.
- Wenn keine eindeutige Bedeutung aus den vorhandenen Informationen
  hervorgeht, sage ausdrücklich, dass die Bedeutung nicht sicher
  bestimmt werden kann.
- Wenn sich bereitgestellte README-Informationen widersprechen,
  vermische die widersprüchlichen Informationen nicht.
- Erfinde keine Quellen.
- Die Antwort soll nicht unnötig lang sein.

Antwortstruktur:
1. Kurze Erklärung des Fachbegriffs.
2. Bedeutung im vorliegenden Arztbrief, sofern der Kontext dafür ausreicht.
3. Nur wenn nötig, Hinweis auf Mehrdeutigkeit oder Unsicherheit.
"""


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
Frage des Nutzers:
{question}

Relevanter Auszug aus dem deutschsprachigen Arztbrief:
{patient_text}

Gefundene README-Einträge aus der kardiologischen Wissensbasis:
{context}

Beantworte die Frage ausschließlich anhand dieser Informationen.
""",
        ),
    ]
)


@dataclass
class RagResult:
    answer: str
    sources: list[dict[str, Any]]
    retrieval_query: str
    local_context: str
    detected_question_terms: list[str]


def calculate_file_hash(path: Path) -> str:
    """Berechnet den SHA-256-Hash einer Datei."""
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def validate_index_state() -> dict[str, Any]:
    """
    Prüft vor dem Laden des FAISS-Index, ob Datenbank,
    Embedding-Modell und Index-Manifest zum aktuellen Systemstand passen.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Die Datenbank wurde nicht gefunden: {DATA_PATH}"
        )

    if not INDEX_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Das Index-Manifest wurde nicht gefunden:\n"
            f"{INDEX_MANIFEST_PATH}\n"
            "Bitte zuerst ausführen: python build_index.py"
        )

    required_index_files = [
        INDEX_DIR / "index.faiss",
        INDEX_DIR / "index.pkl",
    ]

    missing_index_files = [
        path.name for path in required_index_files if not path.exists()
    ]

    if missing_index_files:
        raise FileNotFoundError(
            "Der FAISS-Index ist unvollständig. Es fehlen: "
            + ", ".join(missing_index_files)
            + "\nBitte erneut ausführen: python build_index.py"
        )

    try:
        with open(INDEX_MANIFEST_PATH, "r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Das Index-Manifest konnte nicht korrekt gelesen werden. "
            "Bitte den Index erneut erstellen: python build_index.py"
        ) from exc

    problems: list[str] = []
    current_database_hash = calculate_file_hash(DATA_PATH)

    if manifest.get("database_file") != DATA_PATH.name:
        problems.append(
            "Die Datenbankdatei stimmt nicht mit dem Index-Manifest überein."
        )

    if manifest.get("database_sha256") != current_database_hash:
        problems.append(
            "Der Inhalt der Datenbank wurde seit dem Indexaufbau verändert."
        )

    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        problems.append(
            "Das konfigurierte Embedding-Modell stimmt nicht mit dem für "
            "den Index verwendeten Modell überein."
        )

    if manifest.get("index_unit") != "one_document_per_record_id":
        problems.append(
            "Die Indexeinheit entspricht nicht dem erwarteten Aufbau."
        )

    try:
        database_rows = int(manifest["database_rows"])

        if database_rows < 1:
            raise ValueError

    except (KeyError, TypeError, ValueError):
        problems.append(
            "Die Anzahl der Datenbankzeilen ist im Index-Manifest nicht "
            "gültig dokumentiert."
        )

    if problems:
        raise RuntimeError(
            "Der gespeicherte FAISS-Index passt nicht zum aktuellen System:\n- "
            + "\n- ".join(problems)
            + "\nBitte erneut ausführen: python build_index.py"
        )

    return manifest


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def load_vector_store() -> FAISS:
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Der Index wurde nicht gefunden: {INDEX_DIR}\n"
            "Bitte zuerst ausführen: python build_index.py"
        )

    manifest = validate_index_state()

    # Der Index wird ausschließlich lokal aus dem eigenen Projekt geladen.
    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )

    expected_documents = int(manifest["database_rows"])

    try:
        actual_documents = len(
            vector_store.docstore._dict
        )

    except Exception as exc:
        raise RuntimeError(
            "Die Anzahl der Dokumente im FAISS-Index konnte nicht "
            "überprüft werden."
        ) from exc

    if actual_documents != expected_documents:
        raise RuntimeError(
            "Der FAISS-Index enthält nicht die im Manifest erwartete "
            "Anzahl an Dokumenten.\n"
            f"Erwartet: {expected_documents}\n"
            f"Gefunden: {actual_documents}\n"
            "Bitte erneut ausführen: python build_index.py"
        )

    return vector_store


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )


def normalize_term(value: object) -> str:
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower(),
    )


def term_pattern(term: str) -> str:
    """
    Erstellt ein Regex-Muster für einen vollständigen Begriff.
    """
    return rf"(?<!\w){re.escape(term)}(?!\w)"


def text_contains_term(
    text: str,
    term: str,
    case_sensitive: bool = False,
) -> bool:
    if not text or not term:
        return False

    flags = 0 if case_sensitive else re.IGNORECASE

    return (
        re.search(
            term_pattern(term),
            text,
            flags=flags,
        )
        is not None
    )


def get_all_documents_from_vector_store(
    vector_store: FAISS,
):
    try:
        return list(
            vector_store.docstore._dict.values()
        )

    except Exception as exc:
        raise RuntimeError(
            "Die Dokumente des FAISS-Index konnten nicht gelesen werden."
        ) from exc


def detect_terms_in_question(
    vector_store: FAISS,
    question: str,
) -> list[str]:
    """
    Sucht Begriffe aus der Retrieval-Basis direkt in der Nutzerfrage.

    Die Erkennung dient dazu, den passenden Arztbriefkontext zu bestimmen
    und eindeutige exakte Treffer bei Bedarf zu priorisieren.
    """
    found_terms: list[str] = []
    seen: set[str] = set()

    for doc in get_all_documents_from_vector_store(
        vector_store
    ):
        term_en = str(
            doc.metadata.get(
                "term_en",
                "",
            )
        ).strip()

        term_de = str(
            doc.metadata.get(
                "term_de",
                "",
            )
        ).strip()

        for term in (
            term_de,
            term_en,
        ):
            if not term:
                continue

            # Sehr kurze kleingeschriebene Formen erzeugen
            # zu viele Fehlmatches.
            if (
                len(term) <= 2
                and not term.isupper()
            ):
                continue

            if text_contains_term(
                question,
                term,
            ):
                key = normalize_term(
                    term
                )

                if key not in seen:
                    found_terms.append(
                        term
                    )
                    seen.add(
                        key
                    )

    # Spezifischere/längere Formen stehen zuerst.
    found_terms.sort(
        key=len,
        reverse=True,
    )

    return found_terms


def extract_local_context(
    patient_text: str,
    search_terms: list[str],
    window_chars: int = 700,
) -> str:
    """
    Extrahiert einen lokalen Kontext um das früheste Vorkommen
    eines in der Frage erkannten Zielbegriffs.

    Falls kein Zielbegriff erkannt oder im Arztbrief gefunden wird,
    werden die ersten 1.500 Zeichen als Fallback verwendet.
    """
    patient_text = (
        patient_text or ""
    ).strip()

    if not patient_text:
        return ""

    if not search_terms:
        return patient_text[
            :min(
                len(patient_text),
                1500,
            )
        ]

    matches = []

    for term in sorted(
        search_terms,
        key=len,
        reverse=True,
    ):
        if not term:
            continue

        match = re.search(
            term_pattern(
                term
            ),
            patient_text,
            flags=re.IGNORECASE,
        )

        if match is not None:
            matches.append(
                match
            )

    if not matches:
        return patient_text[
            :min(
                len(patient_text),
                1500,
            )
        ]

    best_match = min(
        matches,
        key=lambda match:
        match.start(),
    )

    start = max(
        0,
        best_match.start()
        - window_chars,
    )

    end = min(
        len(patient_text),
        best_match.end()
        + window_chars,
    )

    previous_break = patient_text.rfind(
        "\n",
        0,
        start,
    )

    if previous_break != -1:
        start = (
            previous_break
            + 1
        )

    next_break = patient_text.find(
        "\n",
        end,
    )

    if next_break != -1:
        end = next_break

    return patient_text[
        start:end
    ].strip()


def exact_term_search(
    vector_store: FAISS,
    detected_terms: list[str],
):
    """
    Liefert alle README-Datensätze,
    deren deutscher oder englischer Term exakt passt.
    """
    if not detected_terms:
        return []

    target_terms = {
        normalize_term(term)
        for term in detected_terms
        if term
    }

    matches = []
    seen_ids: set[str] = set()

    for doc in get_all_documents_from_vector_store(
        vector_store
    ):
        term_norm_en = normalize_term(
            doc.metadata.get(
                "term_norm_en",
                "",
            )
        )

        term_norm_de = normalize_term(
            doc.metadata.get(
                "term_norm_de",
                "",
            )
        )

        if (
            term_norm_en not in target_terms
            and term_norm_de not in target_terms
        ):
            continue

        record_id = str(
            doc.metadata.get(
                "record_id",
                "",
            )
        )

        if record_id in seen_ids:
            continue

        seen_ids.add(
            record_id
        )

        matches.append(
            doc
        )

    return matches


def retrieve_documents(
    vector_store: FAISS,
    retrieval_query: str,
    detected_terms: list[str],
    top_k: int,
):
    """
    Hybrides Retrieval.

    Bei genau einem eindeutigen exakten Treffer wird dieser
    vor den semantischen Treffern priorisiert.

    Bei mehreren exakten Treffern wird keine künstliche Reihenfolge
    vorgegeben. In diesem Fall entscheidet die semantische Ähnlichkeit
    unter Einbezug des lokalen Arztbriefkontexts.

    Diese Trennung ist für mehrdeutige Abkürzungen wie "MI" wichtig.
    """
    if top_k < 1:
        raise ValueError(
            "top_k muss mindestens 1 sein."
        )

    exact_docs = exact_term_search(
        vector_store,
        detected_terms,
    )

    semantic_k = max(
        50,
        top_k * 10,
    )

    semantic_docs = (
        vector_store.similarity_search(
            retrieval_query,
            k=semantic_k,
        )
    )

    # Nur ein wirklich eindeutiger exakter Treffer
    # wird automatisch vorgezogen.
    #
    # Bei mehreren exakten Treffern wird bewusst
    # semantisch anhand von Frage + Arztbriefkontext gerankt.
    if len(exact_docs) == 1:
        candidates = (
            exact_docs
            + semantic_docs
        )

    else:
        candidates = semantic_docs

    combined_docs = []
    seen_ids: set[str] = set()

    for doc in candidates:
        record_id = str(
            doc.metadata.get(
                "record_id",
                "",
            )
        )

        if not record_id:
            record_id = (
                doc.page_content[:100]
            )

        if record_id in seen_ids:
            continue

        seen_ids.add(
            record_id
        )

        combined_docs.append(
            doc
        )

        if (
            len(combined_docs)
            >= top_k
        ):
            break

    return combined_docs


def format_docs(
    docs,
) -> str:
    formatted = []

    for rank, doc in enumerate(
        docs,
        start=1,
    ):
        record_id = doc.metadata.get(
            "record_id",
            "",
        )

        term_en = doc.metadata.get(
            "term_en",
            "",
        )

        term_de = doc.metadata.get(
            "term_de",
            "",
        )

        header = (
            f"[README-Treffer {rank} | "
            f"record_id: {record_id} | "
            f"Deutsch: {term_de or 'nicht vorhanden'} | "
            f"Englisch: {term_en or 'nicht vorhanden'}]"
        )

        formatted.append(
            header
            + "\n"
            + doc.page_content
        )

    return "\n\n---\n\n".join(
        formatted
    )


def extract_sources(
    docs,
) -> list[dict[str, Any]]:
    """
    Bereitet die Retrieval-Treffer
    für Benutzeroberfläche und Evaluation auf.
    """
    sources = []

    for rank, doc in enumerate(
        docs,
        start=1,
    ):
        sources.append(
            {
                "rank":
                    rank,

                "record_id":
                    doc.metadata.get(
                        "record_id",
                        "",
                    ),

                "term_en":
                    doc.metadata.get(
                        "term_en",
                        "",
                    ),

                "term_de":
                    doc.metadata.get(
                        "term_de",
                        "",
                    ),

                "mention_count":
                    doc.metadata.get(
                        "mention_count",
                        0,
                    ),

                "preview":
                    doc.page_content[:900],
            }
        )

    return sources


def detect_database_terms(
    patient_text: str,
    vector_store: FAISS | None = None,
) -> list[dict[str, Any]]:
    """
    Lexikalischer Abgleich des Arztbriefs mit den deutschen
    und englischen Begriffsformen der Retrieval-Basis.

    Diese Funktion kann für die dokumentbezogene
    Terminologieabdeckung in Evaluationsebene 1 verwendet werden.

    Pro normalisierter Begriffsform wird das erste
    Vorkommen im Arztbrief zurückgegeben.
    """
    if not patient_text:
        return []

    if vector_store is None:
        vector_store = (
            load_vector_store()
        )

    candidates: dict[
        str,
        dict[str, Any],
    ] = {}

    for doc in get_all_documents_from_vector_store(
        vector_store
    ):
        record_id = str(
            doc.metadata.get(
                "record_id",
                "",
            )
        )

        term_en = str(
            doc.metadata.get(
                "term_en",
                "",
            )
        ).strip()

        term_de = str(
            doc.metadata.get(
                "term_de",
                "",
            )
        ).strip()

        for term in (
            term_de,
            term_en,
        ):
            if not term:
                continue

            if (
                len(term) <= 2
                and not term.isupper()
            ):
                continue

            key = normalize_term(
                term
            )

            if key not in candidates:
                candidates[key] = {
                    "surface":
                        term,

                    "term_en":
                        term_en,

                    "term_de":
                        term_de,

                    "record_ids":
                        [],
                }

            if (
                record_id
                not in candidates[
                    key
                ][
                    "record_ids"
                ]
            ):
                candidates[
                    key
                ][
                    "record_ids"
                ].append(
                    record_id
                )

    detected = []

    for candidate in candidates.values():
        term = candidate[
            "surface"
        ]

        # Kurze Großbuchstaben-Abkürzungen werden
        # im Arztbrief case-sensitive behandelt.
        case_sensitive = (
            len(term) <= 3
            and term.isupper()
        )

        flags = (
            0
            if case_sensitive
            else re.IGNORECASE
        )

        match = re.search(
            term_pattern(
                term
            ),
            patient_text,
            flags=flags,
        )

        if match is None:
            continue

        detected.append(
            {
                "matched_text":
                    patient_text[
                        match.start():
                        match.end()
                    ],

                "start":
                    match.start(),

                "term_en":
                    candidate[
                        "term_en"
                    ],

                "term_de":
                    candidate[
                        "term_de"
                    ],

                "record_ids":
                    candidate[
                        "record_ids"
                    ],
            }
        )

    detected.sort(
        key=lambda item:
        item["start"]
    )

    return detected


def answer_question(
    question: str,
    patient_text: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> RagResult:
    """
    Führt Retrieval und anschließende
    LLM-Antwortgenerierung aus.
    """
    if top_k < 1:
        raise ValueError(
            "top_k muss mindestens 1 sein."
        )

    vector_store = (
        load_vector_store()
    )

    question = (
        question or ""
    ).strip()

    patient_text = (
        patient_text or ""
    ).strip()

    if not question:
        raise ValueError(
            "Es wurde keine Frage eingegeben."
        )

    detected_question_terms = (
        detect_terms_in_question(
            vector_store,
            question,
        )
    )

    local_context = ""

    if patient_text:
        local_context = (
            extract_local_context(
                patient_text,
                detected_question_terms,
            )
        )

    retrieval_query = question

    if local_context:
        retrieval_query += (
            "\n\n"
            "Relevanter Arztbriefkontext:\n"
            + local_context
        )

    docs = retrieve_documents(
        vector_store=vector_store,
        retrieval_query=retrieval_query,
        detected_terms=detected_question_terms,
        top_k=top_k,
    )

    if not docs:
        raise RuntimeError(
            "Es konnten keine passenden "
            "README-Einträge gefunden werden."
        )

    context = format_docs(
        docs
    )

    if local_context:
        patient_text_for_llm = (
            local_context
        )

    elif patient_text:
        patient_text_for_llm = (
            patient_text[
                :MAX_PATIENT_TEXT_CHARS
            ]
        )

    else:
        patient_text_for_llm = (
            "Kein Arztbriefauszug angegeben."
        )

    chain = (
        RAG_PROMPT
        | get_llm()
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "question":
                question,

            "patient_text":
                patient_text_for_llm,

            "context":
                context,
        }
    )

    return RagResult(
        answer=answer,

        sources=extract_sources(
            docs
        ),

        retrieval_query=
            retrieval_query,

        local_context=
            local_context,

        detected_question_terms=
            detected_question_terms,
    )