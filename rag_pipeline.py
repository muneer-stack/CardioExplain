from __future__ import annotations

import hashlib
import json
import re

from dataclasses import dataclass, field
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
Du bist CardioExplain, ein KI-gestütztes Assistenzsystem zur Erklärung
kardiologischer Fachbegriffe in deutschsprachigen Arztbriefen.

Regeln:
- Antworte immer auf Deutsch.
- Formuliere klar, kurz und laienverständlich auf ungefähr dem Niveau
  der 7. bis 8. Klasse.
- Beantworte genau die gestellte Nutzerfrage. Formuliere keine neue Frage.
- Nutze für medizinische Aussagen ausschließlich die bereitgestellten
  README-Informationen und den Arztbrieftext.
- Ergänze kein medizinisches Wissen aus deinem eigenen Modellwissen.
- Erfinde keine ausgeschriebenen Abkürzungen oder Bedeutungen.
- Gib keine Diagnose, Therapieentscheidung oder individuelle Empfehlung.
- Wenn eine Abkürzung oder ein Begriff mehrere Bedeutungen haben kann, nutze den Arztbriefkontext zur Einordnung.
- Ignoriere README-Treffer, die für die gestellte Frage nicht benötigt werden.

Die Antwort soll kurz und direkt sein.
Antwortstruktur:
1. Kurze Erklärung des Fachbegriffs, soweit die Quellen dies erlauben.
2. Bedeutung im vorliegenden Arztbrief, sofern der Kontext dafür ausreicht.
3. Nur wenn nötig: Hinweis auf fehlende, widersprüchliche oder nicht eindeutige Informationen.
"""


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT,
        ),
        (
            "human",
            """
Frage des Nutzers:
{question}

Technische Abdeckungsprüfung der erfragten Begriffe:
{coverage}

Relevanter deutschsprachiger Arztbrieftext:
{patient_text}

Gefundene README-Einträge aus der kardiologischen Wissensbasis:
{context}

Beantworte die Frage ausschließlich anhand dieser Informationen

Wenn keine ausreichende Wissensgrundlage vorhanden ist, sage dies
klar, anstatt eine Bedeutung zu ergänzen.
""",
        ),
    ]
)


@dataclass
class FocusTarget:
    surface: str
    db_terms: list[str] = field(default_factory=list)
    resolution: str = "unresolved"


@dataclass
class RagResult:
    answer: str
    sources: list[dict[str, Any]]
    retrieval_query: str
    local_context: str
    detected_question_terms: list[str]
    focus_terms: list[str] = field(default_factory=list)
    retrieval_coverage: list[dict[str, Any]] = field(
        default_factory=list
    )


def calculate_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest()


def validate_index_state() -> dict[str, Any]:
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
        path.name
        for path in required_index_files
        if not path.exists()
    ]

    if missing_index_files:
        raise FileNotFoundError(
            "Der FAISS-Index ist unvollständig. Es fehlen: "
            + ", ".join(missing_index_files)
            + "\nBitte erneut ausführen: python build_index.py"
        )

    try:
        with open(
            INDEX_MANIFEST_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            "Das Index-Manifest konnte nicht korrekt gelesen werden. "
            "Bitte den Index erneut erstellen: python build_index.py"
        ) from exc

    problems: list[str] = []

    current_database_hash = calculate_file_hash(
        DATA_PATH
    )

    if manifest.get("database_file") != DATA_PATH.name:
        problems.append(
            "Die Datenbankdatei stimmt nicht mit dem "
            "Index-Manifest überein."
        )

    if (
        manifest.get("database_sha256")
        != current_database_hash
    ):
        problems.append(
            "Der Inhalt der Datenbank wurde seit dem "
            "Indexaufbau verändert."
        )

    if (
        manifest.get("embedding_model")
        != EMBEDDING_MODEL
    ):
        problems.append(
            "Das konfigurierte Embedding-Modell stimmt "
            "nicht mit dem für den Index verwendeten "
            "Modell überein."
        )

    if (
        manifest.get("index_unit")
        != "one_document_per_record_id"
    ):
        problems.append(
            "Die Indexeinheit entspricht nicht dem "
            "erwarteten Aufbau."
        )

    try:
        database_rows = int(
            manifest["database_rows"]
        )

        if database_rows < 1:
            raise ValueError

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        problems.append(
            "Die Anzahl der Datenbankzeilen ist im "
            "Index-Manifest nicht gültig dokumentiert."
        )

    if problems:
        raise RuntimeError(
            "Der gespeicherte FAISS-Index passt nicht "
            "zum aktuellen System:\n- "
            + "\n- ".join(problems)
            + "\nBitte erneut ausführen: "
            "python build_index.py"
        )

    return manifest


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=1)
def load_vector_store() -> FAISS:
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Der Index wurde nicht gefunden: {INDEX_DIR}\n"
            "Bitte zuerst ausführen: python build_index.py"
        )

    manifest = validate_index_state()

    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )

    expected_documents = int(
        manifest["database_rows"]
    )

    try:
        actual_documents = len(
            vector_store.docstore._dict
        )

    except Exception as exc:
        raise RuntimeError(
            "Die Anzahl der Dokumente im FAISS-Index "
            "konnte nicht überprüft werden."
        ) from exc

    if actual_documents != expected_documents:
        raise RuntimeError(
            "Der FAISS-Index enthält nicht die im "
            "Manifest erwartete Anzahl an Dokumenten.\n"
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


def normalize_term(
    value: object,
) -> str:
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower(),
    )


def compact_term(
    value: object,
) -> str:
    text = normalize_term(value)

    return re.sub(
        r"[^a-z0-9äöüß]+",
        "",
        text,
    )


def is_plausible_search_term(
    term: str,
) -> bool:
    if not term:
        return False

    stripped = term.strip()

    if not stripped:
        return False

    if stripped.isdigit():
        return False

    if (
        len(stripped) <= 2
        and not stripped.isupper()
    ):
        return False

    return True


def term_pattern(
    term: str,
) -> str:
    if not term:
        return r"(?!x)x"

    parts = [
        part
        for part in re.split(
            r"[\s\-‐-‒–—]+",
            term.strip(),
        )
        if part
    ]

    if not parts:
        return r"(?!x)x"

    escaped_parts = [
        re.escape(part)
        for part in parts
    ]

    separator = (
        r"(?:"
        r"\s+"
        r"|"
        r"\s*[-‐-‒–—]\s*"
        r")"
    )

    core = separator.join(
        escaped_parts
    )

    return rf"(?<!\w){core}(?!\w)"


def text_contains_term(
    text: str,
    term: str,
    case_sensitive: bool = False,
) -> bool:
    if not text or not term:
        return False

    flags = (
        0
        if case_sensitive
        else re.IGNORECASE
    )

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
            "Die Dokumente des FAISS-Index "
            "konnten nicht gelesen werden."
        ) from exc


def get_record_id(
    doc,
) -> str:
    record_id = str(
        doc.metadata.get(
            "record_id",
            "",
        )
    ).strip()

    if record_id:
        return record_id

    return doc.page_content[:100]


def detect_terms_in_question(
    vector_store: FAISS,
    question: str,
) -> list[str]:
    if not question:
        return []

    found: dict[
        str,
        tuple[int, str],
    ] = {}

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
            if not is_plausible_search_term(
                term
            ):
                continue

            match = re.search(
                term_pattern(term),
                question,
                flags=re.IGNORECASE,
            )

            if match is None:
                continue

            key = normalize_term(term)

            current = found.get(key)

            candidate = (
                match.start(),
                term,
            )

            if (
                current is None
                or candidate[0]
                < current[0]
            ):
                found[key] = candidate

    ordered = sorted(
        found.values(),
        key=lambda item: (
            item[0],
            -len(item[1]),
        ),
    )

    return [
        term
        for _, term in ordered
    ]


FOCUS_TRIGGER_PATTERN = re.compile(
    r"(?:"
    r"was\s+bedeutet"
    r"|was\s+heißt"
    r"|was\s+heisst"
    r"|wofür\s+steht"
    r"|erkläre"
    r"|erklären\s+sie"
    r"|bedeutung\s+von"
    r"|was\s+ist"
    r")",
    flags=re.IGNORECASE,
)


def extract_focus_text(
    question: str,
) -> str:
    if not question:
        return ""

    matches = list(
        FOCUS_TRIGGER_PATTERN.finditer(
            question
        )
    )

    if not matches:
        return question.strip()

    last_match = matches[-1]

    focus_text = question[
        last_match.end():
    ]

    return focus_text.strip(
        " \t\n:,-"
    )


def extract_special_surface_tokens(
    text: str,
) -> list[str]:
    if not text:
        return []

    patterns = [
        r"\b[A-ZÄÖÜ][A-ZÄÖÜ0-9]{1,10}\b",
        r"\b[A-Za-zÄÖÜäöüß0-9]+"
        r"(?:[-/][A-Za-zÄÖÜäöüß0-9]+)+\b",
        r"\b[A-Za-zÄÖÜäöüß]+"
        r"[A-ZÄÖÜ]"
        r"[A-Za-zÄÖÜäöüß0-9-]*\b",
    ]

    found: list[
        tuple[int, str]
    ] = []

    seen: set[str] = set()

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            text,
        ):
            surface = match.group(0)

            key = normalize_term(
                surface
            )

            if key in seen:
                continue

            seen.add(key)

            found.append(
                (
                    match.start(),
                    surface,
                )
            )

    found.sort(
        key=lambda item: item[0]
    )

    return [
        surface
        for _, surface in found
    ]


def get_known_database_terms(
    vector_store: FAISS,
) -> list[str]:
    terms: dict[str, str] = {}

    for doc in get_all_documents_from_vector_store(
        vector_store
    ):
        for field_name in (
            "term_de",
            "term_en",
        ):
            term = str(
                doc.metadata.get(
                    field_name,
                    "",
                )
            ).strip()

            if not is_plausible_search_term(
                term
            ):
                continue

            key = normalize_term(term)

            if key not in terms:
                terms[key] = term

    return list(
        terms.values()
    )


def resolve_surface_to_database_terms(
    surface: str,
    known_terms: list[str],
) -> tuple[
    list[str],
    str,
]:
    surface_norm = normalize_term(
        surface
    )

    surface_compact = compact_term(
        surface
    )

    if not surface_norm:
        return [], "unresolved"

    exact = []

    for term in known_terms:
        if (
            normalize_term(term)
            == surface_norm
        ):
            exact.append(term)

    if exact:
        return (
            _unique_terms(exact),
            "exact",
        )

    embedded_candidates: list[
        tuple[int, str]
    ] = []

    if len(surface_compact) >= 4:
        for term in known_terms:
            term_compact = compact_term(
                term
            )

            if len(term_compact) < 4:
                continue

            if (
                term_compact
                == surface_compact
            ):
                continue

            if (
                term_compact
                in surface_compact
            ):
                embedded_candidates.append(
                    (
                        len(term_compact),
                        term,
                    )
                )

    if not embedded_candidates:
        return [], "unresolved"

    max_length = max(
        length
        for length, _
        in embedded_candidates
    )

    longest = [
        term
        for length, term
        in embedded_candidates
        if length == max_length
    ]

    return (
        _unique_terms(longest),
        "derived",
    )


def _unique_terms(
    terms: list[str],
) -> list[str]:
    result = []
    seen = set()

    for term in terms:
        key = normalize_term(
            term
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        result.append(term)

    return result


def build_focus_targets(
    vector_store: FAISS,
    question: str,
) -> list[FocusTarget]:
    focus_text = extract_focus_text(
        question
    )

    known_terms = get_known_database_terms(
        vector_store
    )

    candidate_surfaces: list[str] = []

    direct_focus_terms = (
        detect_terms_in_question(
            vector_store,
            focus_text,
        )
    )

    candidate_surfaces.extend(
        direct_focus_terms
    )

    candidate_surfaces.extend(
        extract_special_surface_tokens(
            focus_text
        )
    )

    if not candidate_surfaces:
        candidate_surfaces.extend(
            detect_terms_in_question(
                vector_store,
                question,
            )
        )

    candidate_surfaces = _unique_terms(
        candidate_surfaces
    )

    targets: list[
        FocusTarget
    ] = []

    for surface in candidate_surfaces:
        db_terms, resolution = (
            resolve_surface_to_database_terms(
                surface=surface,
                known_terms=known_terms,
            )
        )

        targets.append(
            FocusTarget(
                surface=surface,
                db_terms=db_terms,
                resolution=resolution,
            )
        )

    return targets


WORD_PATTERN = re.compile(
    r"[A-Za-zÄÖÜäöüß0-9]+"
    r"(?:[-/][A-Za-zÄÖÜäöüß0-9]+)*"
)


def find_best_question_span(
    patient_text: str,
    question: str,
    focus_surfaces: list[str],
    min_words: int = 4,
    max_words: int = 18,
):
    if not patient_text or not question:
        return None

    words = WORD_PATTERN.findall(
        question
    )

    if len(words) < min_words:
        return None

    focus_compact = {
        compact_term(surface)
        for surface in focus_surfaces
        if surface
    }

    max_n = min(
        max_words,
        len(words),
    )

    for n in range(
        max_n,
        min_words - 1,
        -1,
    ):
        for start_index in range(
            0,
            len(words) - n + 1,
        ):
            window = words[
                start_index:
                start_index + n
            ]

            if focus_compact:
                window_compact = {
                    compact_term(word)
                    for word in window
                }

                contains_focus = any(
                    focus in window_compact
                    or any(
                        focus in word
                        or word in focus
                        for word in window_compact
                        if len(word) >= 3
                    )
                    for focus in focus_compact
                    if focus
                )

                if not contains_focus:
                    continue

            escaped_words = [
                re.escape(word)
                for word in window
            ]

            separator = (
                r"(?:"
                r"\s+"
                r"|"
                r"[\s,.;:()/_\-‐-‒–—]+"
                r")"
            )

            pattern = separator.join(
                escaped_words
            )

            match = re.search(
                pattern,
                patient_text,
                flags=re.IGNORECASE,
            )

            if match is not None:
                return match

    return None


def extract_local_context(
    patient_text: str,
    search_terms: list[str],
    question: str = "",
    focus_surfaces: list[str] | None = None,
    window_chars: int = 700,
) -> str:
    patient_text = (
        patient_text or ""
    ).strip()

    if not patient_text:
        return ""

    focus_surfaces = (
        focus_surfaces
        or []
    )

    question_match = (
        find_best_question_span(
            patient_text=patient_text,
            question=question,
            focus_surfaces=focus_surfaces,
        )
    )

    if question_match is not None:
        return _context_around_match(
            patient_text=patient_text,
            match=question_match,
            window_chars=window_chars,
        )

    ordered_terms = _unique_terms(
        focus_surfaces
        + search_terms
    )

    for term in ordered_terms:
        if not term:
            continue

        match = re.search(
            term_pattern(term),
            patient_text,
            flags=re.IGNORECASE,
        )

        if match is not None:
            return _context_around_match(
                patient_text=patient_text,
                match=match,
                window_chars=window_chars,
            )

    return patient_text[
        :min(
            len(patient_text),
            1500,
        )
    ]


def _context_around_match(
    patient_text: str,
    match,
    window_chars: int,
) -> str:
    start = max(
        0,
        match.start()
        - window_chars,
    )

    end = min(
        len(patient_text),
        match.end()
        + window_chars,
    )

    previous_break = (
        patient_text.rfind(
            "\n",
            0,
            start,
        )
    )

    if previous_break != -1:
        start = (
            previous_break
            + 1
        )

    next_break = (
        patient_text.find(
            "\n",
            end,
        )
    )

    if next_break != -1:
        end = next_break

    return patient_text[
        start:end
    ].strip()


def exact_term_search_for_term(
    vector_store: FAISS,
    term: str,
):
    target = normalize_term(
        term
    )

    if not target:
        return []

    matches = []
    seen_ids: set[str] = set()

    for doc in get_all_documents_from_vector_store(
        vector_store
    ):
        term_norm_en = normalize_term(
            doc.metadata.get(
                "term_norm_en",
                doc.metadata.get(
                    "term_en",
                    "",
                ),
            )
        )

        term_norm_de = normalize_term(
            doc.metadata.get(
                "term_norm_de",
                doc.metadata.get(
                    "term_de",
                    "",
                ),
            )
        )

        if (
            term_norm_en != target
            and term_norm_de != target
        ):
            continue

        record_id = get_record_id(
            doc
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


def exact_term_search(
    vector_store: FAISS,
    detected_terms: list[str],
):
    matches = []
    seen_ids: set[str] = set()

    for term in detected_terms:
        term_matches = (
            exact_term_search_for_term(
                vector_store,
                term,
            )
        )

        for doc in term_matches:
            record_id = get_record_id(
                doc
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


def build_semantic_rank(
    semantic_docs,
) -> dict[str, int]:
    result: dict[
        str,
        int,
    ] = {}

    for rank, doc in enumerate(
        semantic_docs
    ):
        record_id = get_record_id(
            doc
        )

        if record_id not in result:
            result[
                record_id
            ] = rank

    return result


def sort_exact_candidates(
    docs,
    semantic_rank: dict[str, int],
):
    large_rank = 10**9

    return sorted(
        docs,
        key=lambda doc: (
            semantic_rank.get(
                get_record_id(doc),
                large_rank,
            ),
            -int(
                doc.metadata.get(
                    "mention_count",
                    0,
                )
                or 0
            ),
        ),
    )


def retrieve_documents(
    vector_store: FAISS,
    retrieval_query: str,
    detected_terms: list[str],
    top_k: int,
    focus_targets: list[FocusTarget] | None = None,
):
    if top_k < 1:
        raise ValueError(
            "top_k muss mindestens 1 sein."
        )

    focus_targets = (
        focus_targets
        or []
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

    semantic_rank = build_semantic_rank(
        semantic_docs
    )

    selected_docs = []
    seen_ids: set[str] = set()

    for target in focus_targets:
        if len(
            selected_docs
        ) >= top_k:
            break

        if not target.db_terms:
            continue

        target_candidates = []
        target_seen = set()

        for db_term in target.db_terms:
            exact_docs = (
                exact_term_search_for_term(
                    vector_store,
                    db_term,
                )
            )

            for doc in exact_docs:
                record_id = (
                    get_record_id(
                        doc
                    )
                )

                if (
                    record_id
                    in target_seen
                ):
                    continue

                target_seen.add(
                    record_id
                )

                target_candidates.append(
                    doc
                )

        if not target_candidates:
            continue

        target_candidates = (
            sort_exact_candidates(
                target_candidates,
                semantic_rank,
            )
        )

        for doc in target_candidates:
            record_id = get_record_id(
                doc
            )

            if record_id in seen_ids:
                continue

            seen_ids.add(
                record_id
            )

            selected_docs.append(
                doc
            )

            break

    if not focus_targets:
        exact_docs = exact_term_search(
            vector_store,
            detected_terms,
        )

        exact_docs = sort_exact_candidates(
            exact_docs,
            semantic_rank,
        )

        for doc in exact_docs:
            if (
                len(selected_docs)
                >= top_k
            ):
                break

            record_id = get_record_id(
                doc
            )

            if record_id in seen_ids:
                continue

            seen_ids.add(
                record_id
            )

            selected_docs.append(
                doc
            )

    for doc in semantic_docs:
        if (
            len(selected_docs)
            >= top_k
        ):
            break

        record_id = get_record_id(
            doc
        )

        if record_id in seen_ids:
            continue

        seen_ids.add(
            record_id
        )

        selected_docs.append(
            doc
        )

    return selected_docs


def build_retrieval_coverage(
    vector_store: FAISS,
    focus_targets: list[FocusTarget],
    selected_docs,
) -> list[dict[str, Any]]:
    selected_ids = {
        get_record_id(doc)
        for doc in selected_docs
    }

    coverage = []

    for target in focus_targets:
        exact_ids = []

        for db_term in target.db_terms:
            for doc in (
                exact_term_search_for_term(
                    vector_store,
                    db_term,
                )
            ):
                record_id = (
                    get_record_id(
                        doc
                    )
                )

                if (
                    record_id
                    not in exact_ids
                ):
                    exact_ids.append(
                        record_id
                    )

        selected_for_target = [
            record_id
            for record_id in exact_ids
            if record_id in selected_ids
        ]

        if target.resolution == "exact":
            status = (
                "Exakte README-Begriffsform gefunden."
            )

        elif target.resolution == "derived":
            status = (
                "Die genaue Form fehlt in README. "
                "Eine vorhandene Grundform wurde "
                "automatisch erkannt."
            )

        else:
            status = (
                "Keine passende README-Begriffsform "
                "oder automatisch erkennbare Grundform "
                "gefunden."
            )

        coverage.append(
            {
                "surface":
                    target.surface,

                "resolution":
                    target.resolution,

                "database_terms":
                    list(
                        target.db_terms
                    ),

                "exact_record_ids":
                    exact_ids,

                "selected_record_ids":
                    selected_for_target,

                "status":
                    status,
            }
        )

    return coverage


def format_coverage_for_prompt(
    coverage: list[dict[str, Any]],
) -> str:
    if not coverage:
        return (
            "Es konnte kein eindeutiger Zielbegriff "
            "automatisch bestimmt werden. "
            "Die ausgewählten README-Treffer müssen "
            "deshalb besonders vorsichtig anhand der "
            "Frage geprüft werden."
        )

    lines = []

    for item in coverage:
        surface = item.get(
            "surface",
            "",
        )

        resolution = item.get(
            "resolution",
            "",
        )

        database_terms = (
            item.get(
                "database_terms",
                [],
            )
        )

        selected_ids = (
            item.get(
                "selected_record_ids",
                [],
            )
        )

        lines.append(
            f"Zielbegriff aus der Frage: {surface}"
        )

        if resolution == "exact":
            lines.append(
                "- Die Begriffsform ist exakt "
                "in README vorhanden."
            )

        elif resolution == "derived":
            lines.append(
                "- Die genaue Begriffsform ist nicht "
                "direkt in README vorhanden."
            )

            if database_terms:
                lines.append(
                    "- Automatisch erkannte "
                    "README-Grundform: "
                    + ", ".join(
                        database_terms
                    )
                )

            lines.append(
                "- Aus der Grundform dürfen keine "
                "nicht belegten Zusätze oder "
                "Abkürzungsbestandteile erklärt werden."
            )

        else:
            lines.append(
                "- Keine passende Begriffsform oder "
                "Grundform wurde in README gefunden."
            )

            lines.append(
                "- Keine sichere Bedeutung aus "
                "thematisch ähnlichen Treffern ableiten."
            )

        if selected_ids:
            lines.append(
                "- Passender formaler README-Kandidat "
                "wurde in die Top-Treffer übernommen: "
                + ", ".join(
                    selected_ids
                )
            )

        elif (
            resolution
            in {
                "exact",
                "derived",
            }
        ):
            lines.append(
                "- Eine bekannte Form existiert, wurde "
                "aber nicht als formaler Zieltreffer "
                "in die ausgewählten Treffer übernommen."
            )

        lines.append(
            "- Achtung: Eine gleiche Form beweist "
            "nicht automatisch eine zum Arztbrief "
            "passende medizinische Bedeutung."
        )

        lines.append("")

    return "\n".join(
        lines
    ).strip()


def format_docs(
    docs,
) -> str:
    formatted = []

    for rank, doc in enumerate(
        docs,
        start=1,
    ):
        record_id = (
            doc.metadata.get(
                "record_id",
                "",
            )
        )

        term_en = (
            doc.metadata.get(
                "term_en",
                "",
            )
        )

        term_de = (
            doc.metadata.get(
                "term_de",
                "",
            )
        )

        header = (
            f"[README-Treffer {rank} | "
            f"record_id: {record_id} | "
            f"Deutsch: "
            f"{term_de or 'nicht vorhanden'} | "
            f"Englisch: "
            f"{term_en or 'nicht vorhanden'}]"
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
                    doc.page_content,
            }
        )

    return sources


def detect_database_terms(
    patient_text: str,
    vector_store: FAISS | None = None,
) -> list[dict[str, Any]]:
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
            if not is_plausible_search_term(
                term
            ):
                continue

            key = normalize_term(
                term
            )

            if key not in candidates:
                candidates[
                    key
                ] = {
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

    for candidate in (
        candidates.values()
    ):
        term = candidate[
            "surface"
        ]

        case_sensitive = (
            len(term) <= 2
            and term.isupper()
        )

        flags = (
            0
            if case_sensitive
            else re.IGNORECASE
        )

        match = re.search(
            term_pattern(term),
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

                "end":
                    match.end(),

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
        key=lambda item: (
            item["start"],
            -(
                item["end"]
                - item["start"]
            ),
        )
    )

    return detected


def build_retrieval_query(
    question: str,
    focus_targets: list[FocusTarget],
    local_context: str,
) -> str:
    parts = []

    focus_surfaces = [
        target.surface
        for target in focus_targets
        if target.surface
    ]

    resolved_terms = []

    for target in focus_targets:
        resolved_terms.extend(
            target.db_terms
        )

    resolved_terms = _unique_terms(
        resolved_terms
    )

    if focus_surfaces:
        parts.append(
            "Gesuchte Begriffe aus der Nutzerfrage:\n"
            + ", ".join(
                focus_surfaces
            )
        )

    if resolved_terms:
        parts.append(
            "Automatisch zugeordnete "
            "README-Begriffsformen:\n"
            + ", ".join(
                resolved_terms
            )
        )

    parts.append(
        "Nutzerfrage:\n"
        + question
    )

    if local_context:
        parts.append(
            "Zielnaher Arztbriefkontext:\n"
            + local_context
        )

    return "\n\n".join(
        parts
    )


def build_patient_text_for_llm(
    patient_text: str,
    local_context: str,
) -> str:
    patient_text = (
        patient_text or ""
    ).strip()

    local_context = (
        local_context or ""
    ).strip()

    if not patient_text:
        return (
            "Kein Arztbriefauszug angegeben."
        )

    if (
        len(patient_text)
        <= MAX_PATIENT_TEXT_CHARS
    ):
        if (
            local_context
            and local_context
            != patient_text
        ):
            return (
                "Zielnaher Ausschnitt:\n"
                + local_context
                + "\n\n"
                + "Vollständiger verfügbarer "
                "Arztbrieftext:\n"
                + patient_text
            )

        return patient_text

    if local_context:
        return local_context[
            :MAX_PATIENT_TEXT_CHARS
        ]

    return patient_text[
        :MAX_PATIENT_TEXT_CHARS
    ]


def answer_question(
    question: str,
    patient_text: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> RagResult:
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

    focus_targets = (
        build_focus_targets(
            vector_store,
            question,
        )
    )

    focus_surfaces = [
        target.surface
        for target in focus_targets
        if target.surface
    ]

    resolved_focus_terms = []

    for target in focus_targets:
        resolved_focus_terms.extend(
            target.db_terms
        )

    resolved_focus_terms = (
        _unique_terms(
            resolved_focus_terms
        )
    )

    local_context = ""

    if patient_text:
        context_search_terms = (
            focus_surfaces
            + resolved_focus_terms
            + detected_question_terms
        )

        context_search_terms = (
            _unique_terms(
                context_search_terms
            )
        )

        local_context = (
            extract_local_context(
                patient_text=patient_text,
                search_terms=(
                    context_search_terms
                ),
                question=question,
                focus_surfaces=(
                    focus_surfaces
                ),
            )
        )

    retrieval_query = (
        build_retrieval_query(
            question=question,
            focus_targets=focus_targets,
            local_context=local_context,
        )
    )

    docs = retrieve_documents(
        vector_store=vector_store,
        retrieval_query=retrieval_query,
        detected_terms=(
            detected_question_terms
        ),
        top_k=top_k,
        focus_targets=focus_targets,
    )

    if not docs:
        raise RuntimeError(
            "Es konnten keine README-Einträge "
            "gefunden werden."
        )

    retrieval_coverage = (
        build_retrieval_coverage(
            vector_store=vector_store,
            focus_targets=focus_targets,
            selected_docs=docs,
        )
    )

    coverage_text = (
        format_coverage_for_prompt(
            retrieval_coverage
        )
    )

    context = format_docs(
        docs
    )

    patient_text_for_llm = (
        build_patient_text_for_llm(
            patient_text=patient_text,
            local_context=local_context,
        )
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

            "coverage":
                coverage_text,

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

        focus_terms=[
            target.surface
            for target
            in focus_targets
        ],

        retrieval_coverage=
            retrieval_coverage,
    )
