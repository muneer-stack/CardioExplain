from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from config import (
    DATA_PATH,
    INDEX_DIR,
    DEFAULT_TOP_K,
)

from rag_pipeline import (
    answer_question,
    detect_database_terms,
)


st.set_page_config(
    page_title="CardioExplain RAG",
    page_icon="🫀",
    layout="wide",
)


def read_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    if suffix == ".txt":
        return uploaded_file.getvalue().decode(
            "utf-8",
            errors="ignore",
        )

    if suffix == ".pdf":
        pdf_bytes = uploaded_file.getvalue()

        reader = PdfReader(
            io.BytesIO(pdf_bytes)
        )

        pages = []

        for page in reader.pages:
            pages.append(
                page.extract_text() or ""
            )

        return "\n".join(pages)

    return ""


@st.cache_data
def load_database_preview():
    if not DATA_PATH.exists():
        return {
            "rows": 0,
            "unique_de": 0,
            "unique_en": 0,
            "mention_count": 0,
        }

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        encoding="utf-8",
    )

    unique_de = (
        df["ann_text_de"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .str.lower()
        .nunique()
    )

    unique_en = (
        df["ann_text_en"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .str.lower()
        .nunique()
    )

    mention_count = (
        pd.to_numeric(
            df["mention_count"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    return {
        "rows": len(df),
        "unique_de": unique_de,
        "unique_en": unique_en,
        "mention_count": int(
            mention_count
        ),
    }


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hallo! Ich bin CardioExplain. "
                    "Lade einen deutschsprachigen Arztbrief hoch "
                    "oder füge einen Auszug ein. "
                    "Danach kannst du Fragen zu kardiologischen "
                    "Fachbegriffen stellen."
                ),
            }
        ]

    if "patient_text" not in st.session_state:
        st.session_state.patient_text = ""


def sidebar() -> None:
    st.sidebar.title(
        "System"
    )

    if INDEX_DIR.exists():
        st.sidebar.success(
            "FAISS-Index gefunden"
        )
    else:
        st.sidebar.error(
            "FAISS-Index fehlt. "
            "Bitte zuerst `python build_index.py` ausführen."
        )

    database_info = (
        load_database_preview()
    )

    st.sidebar.markdown(
        "### README-Wissensbasis"
    )

    st.sidebar.write(
        f"Datensätze: "
        f"**{database_info['rows']:,}**"
    )

    st.sidebar.write(
        f"Deutsche Begriffsformen: "
        f"**{database_info['unique_de']:,}**"
    )

    st.sidebar.write(
        f"Englische Begriffsformen: "
        f"**{database_info['unique_en']:,}**"
    )

    st.sidebar.write(
        f"Ursprüngliche Erwähnungen: "
        f"**{database_info['mention_count']:,}**"
    )

    st.sidebar.write(
        f"README-Treffer pro Frage: "
        f"**{DEFAULT_TOP_K}**"
    )

    st.sidebar.markdown(
        "### Hinweis"
    )

    st.sidebar.info(
        "CardioExplain ist ein Forschungsprototyp "
        "und ersetzt keine medizinische Beratung."
    )


def show_detected_terms(
    patient_text: str,
) -> None:
    if not patient_text.strip():
        st.warning(
            "Bitte zuerst einen Arztbrief "
            "oder einen Textauszug eingeben."
        )
        return

    with st.spinner(
        "Suche Begriffe aus der README-Wissensbasis ..."
    ):
        detected = (
            detect_database_terms(
                patient_text
            )
        )

    if not detected:
        st.info(
            "Es wurden keine wörtlich übereinstimmenden "
            "Begriffe aus der README-Wissensbasis gefunden."
        )
        return

    rows = []
    seen = set()

    for item in detected:
        matched = str(
            item.get(
                "matched_text",
                "",
            )
        ).strip()

        key = matched.lower()

        if key in seen:
            continue

        seen.add(key)

        rows.append(
            {
                "Im Arztbrief erkannt":
                    matched,

                "Deutsch":
                    item.get(
                        "term_de",
                        "",
                    ),

                "Englisch":
                    item.get(
                        "term_en",
                        "",
                    ),

                "README-record_ids":
                    " | ".join(
                        item.get(
                            "record_ids",
                            [],
                        )
                    ),
            }
        )

    result_df = pd.DataFrame(
        rows
    )

    st.success(
        f"{len(result_df)} unterschiedliche "
        "README-Begriffe erkannt."
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Die Erkennung basiert auf wörtlichen Übereinstimmungen "
        "mit den deutschen und englischen Begriffsformen "
        "der verwendeten README-Wissensbasis."
    )


def main() -> None:
    init_state()
    sidebar()

    st.title(
        "CardioExplain"
    )

    st.caption(
        "KI-gestütztes RAG-System zur "
        "laienverständlichen Erklärung "
        "kardiologischer Fachbegriffe "
        "in deutschsprachigen Arztbriefen."
    )

    with st.expander(
        "Arztbrief hochladen oder Text einfügen",
        expanded=True,
    ):
        uploaded_file = (
            st.file_uploader(
                "PDF oder TXT hochladen",
                type=[
                    "pdf",
                    "txt",
                ],
            )
        )

        if uploaded_file is not None:
            extracted_text = (
                read_uploaded_file(
                    uploaded_file
                )
            )

            if extracted_text:
                st.session_state.patient_text = (
                    extracted_text
                )

                st.success(
                    "Text wurde aus der Datei gelesen."
                )

            else:
                st.warning(
                    "Aus dieser Datei konnte "
                    "kein Text gelesen werden."
                )

        pasted_text = st.text_area(
            "Oder Arztbrief-Auszug hier einfügen",
            value=st.session_state.patient_text,
            height=260,
            placeholder=(
                "Hier kann ein deutschsprachiger "
                "Arztbrief oder ein relevanter "
                "Ausschnitt eingefügt werden."
            ),
        )

        st.session_state.patient_text = (
            pasted_text
        )

        if st.button(
            "Wörtlich erkannte README-Begriffe anzeigen"
        ):
            show_detected_terms(
                st.session_state.patient_text
            )

    with st.expander(
        "Beispielfragen"
    ):
        st.markdown(
            """
- Was bedeutet Vorhofflimmern?
- Was bedeutet Mitralinsuffizienz?
- Was bedeutet MI in diesem Arztbrief?
- Was bedeutet EKG?
"""
        )

    st.divider()

    for message in (
        st.session_state.messages
    ):
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    user_question = st.chat_input(
        "Frage zu einem kardiologischen Fachbegriff stellen ..."
    )

    if user_question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        with st.chat_message(
            "user"
        ):
            st.markdown(
                user_question
            )

        with st.chat_message(
            "assistant"
        ):
            with st.spinner(
                "Suche in README und generiere Antwort ..."
            ):
                try:
                    result = (
                        answer_question(
                            question=user_question,
                            patient_text=(
                                st.session_state.patient_text
                            ),
                            top_k=DEFAULT_TOP_K,
                        )
                    )

                    st.markdown(
                        result.answer
                    )

                    with st.expander(
                        "Genutzte README-Treffer"
                    ):
                        for source in (
                            result.sources
                        ):
                            rank = source.get(
                                "rank",
                                "",
                            )

                            record_id = (
                                source.get(
                                    "record_id",
                                    "",
                                )
                            )

                            term_de = (
                                source.get(
                                    "term_de",
                                    "",
                                )
                            )

                            term_en = (
                                source.get(
                                    "term_en",
                                    "",
                                )
                            )

                            mention_count = (
                                source.get(
                                    "mention_count",
                                    0,
                                )
                            )

                            st.markdown(
                                f"**{rank}. "
                                f"{term_de or term_en}**"
                            )

                            st.caption(
                                f"record_id: {record_id} | "
                                f"Englisch: {term_en or '-'} | "
                                f"mention_count: {mention_count}"
                            )

                            st.code(
                                source.get(
                                    "preview",
                                    "",
                                )
                            )

                    with st.expander(
                        "Retrieval-Details"
                    ):
                        st.markdown(
                            "**In der Frage erkannte README-Begriffe:**"
                        )

                        if result.detected_question_terms:
                            st.write(
                                ", ".join(
                                    result.detected_question_terms
                                )
                            )
                        else:
                            st.write(
                                "Kein exakter README-Begriff erkannt."
                            )

                        st.markdown(
                            "**Verwendeter lokaler Arztbriefkontext:**"
                        )

                        st.code(
                            result.local_context
                            or
                            "Kein lokaler Kontext verfügbar."
                        )

                        st.markdown(
                            "**Retrieval-Anfrage:**"
                        )

                        st.code(
                            result.retrieval_query
                        )

                    st.session_state.messages.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                result.answer,
                        }
                    )

                except Exception as exc:
                    error_msg = (
                        "Es ist ein Fehler aufgetreten. "
                        "Prüfe bitte, ob der Index erstellt wurde "
                        "und ob Ollama läuft.\n\n"
                        f"Fehler: `{exc}`"
                    )

                    st.error(
                        error_msg
                    )

                    st.session_state.messages.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                error_msg,
                        }
                    )


if __name__ == "__main__":
    main()