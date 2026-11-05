"""Ask Documents page. A functional page for Phase 4; Phase 6 replaces its layout."""

from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from core.extractive import GenerationMethod
from core.question_answering import Answer, QuestionAnswerer
from database.db_manager import DatabaseManager
from database.schema import DocumentStatus

_ALL_DOCUMENTS = ""
_RECENT_QUESTIONS = 4
_METHOD_LABELS = {
    GenerationMethod.LLM: "Local LLM answer",
    GenerationMethod.EXTRACTIVE: "Extractive NLP (not AI-written)",
    GenerationMethod.NONE: "No answer",
}


def _payload(answer: Answer) -> dict:
    return {
        "method": answer.method.value,
        "model": answer.model,
        "notices": list(answer.notices),
        "sources": [asdict(source) for source in answer.sources],
    }


def _show_assistant(content: str, payload: dict) -> None:
    method = GenerationMethod(payload["method"])
    label = _METHOD_LABELS[method]
    st.caption(f"{label}: {payload['model']}" if payload.get("model") else label)
    st.markdown(content)
    for notice in payload.get("notices", []):
        st.warning(notice)
    sources = payload.get("sources", [])
    if sources:
        with st.expander(f"Evidence from the documents ({len(sources)})"):
            for source in sources:
                page = source["page_number"]
                place = f"{source['filename']}, p. {page}" if page else source["filename"]
                state = "cited" if source["cited"] else "retrieved"
                st.markdown(
                    f"**{source['label']}: {place}** ({state}, similarity {source['score']:.2f})"
                )
                st.text(source["text"])


def render(db: DatabaseManager, answerer: QuestionAnswerer, model: str | None) -> None:
    st.subheader("Ask your documents")
    documents = db.list_by_status(DocumentStatus.READY)
    if not documents:
        st.caption("Upload a document first.")
        return

    names = {_ALL_DOCUMENTS: "All documents"}
    names |= {doc.document_id: doc.original_filename for doc in documents}
    selected = st.selectbox("Search in", list(names), format_func=names.get)
    document_id = selected or None

    conversation_id = db.latest_conversation(document_id)
    left, right = st.columns(2)
    if left.button("New conversation"):
        db.create_conversation(document_id)
        st.rerun()
    if right.button("Clear conversation", disabled=conversation_id is None):
        db.delete_conversation(conversation_id)
        st.rerun()

    messages = db.list_messages(conversation_id) if conversation_id else []
    for message in messages:
        with st.chat_message(message.role):
            if message.role == "user":
                st.markdown(message.content)
            else:
                _show_assistant(message.content, message.payload)

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return
    if conversation_id is None:
        conversation_id = db.create_conversation(document_id)
    with st.spinner("Searching the documents"):
        answer = answerer.ask(
            question,
            [document_id] if document_id else None,
            [m.content for m in messages if m.role == "user"][-_RECENT_QUESTIONS:],
            model=model,
            use_llm=model is not None,
        )
    db.add_message(conversation_id, "user", question)
    db.add_message(conversation_id, "assistant", answer.text, _payload(answer))
    st.rerun()
