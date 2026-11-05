"""Ask Documents page."""

from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from core.extractive import GenerationMethod
from core.question_answering import Answer
from database.schema import DocumentStatus
from ui import components as ui

_ALL_DOCUMENTS = ""
_RECENT_QUESTIONS = 4
_PENDING_KEY = "pending_question"
_SUGGESTIONS = (
    "What are the main objectives?",
    "What are the important findings?",
    "What methodology was used?",
    "What limitations are mentioned?",
)
_METHOD_PILLS = {
    GenerationMethod.LLM: ("Local LLM", "accent"),
    GenerationMethod.EXTRACTIVE: ("Extractive NLP, not AI-written", "warn"),
    GenerationMethod.NONE: ("No answer", "neutral"),
}


def _payload(answer: Answer) -> dict:
    return {
        "method": answer.method.value,
        "model": answer.model,
        "notices": list(answer.notices),
        "sources": [asdict(source) for source in answer.sources],
    }


def _show_assistant(content: str, payload: dict) -> None:
    label, tone = _METHOD_PILLS[GenerationMethod(payload["method"])]
    if payload.get("model") and payload["method"] == GenerationMethod.LLM.value:
        label = f"{label}: {payload['model']}"
    ui.pill(label, tone)
    st.markdown(ui.safe_markdown(content))
    for notice in payload.get("notices", []):
        st.caption(ui.safe_markdown(notice))
    sources = payload.get("sources", [])
    if sources:
        cited = sum(1 for source in sources if source["cited"])
        title = f"Sources: {cited} cited, {len(sources)} retrieved" if cited else (
            f"Sources: {len(sources)} retrieved"
        )
        with st.expander(title):
            for source in sources:
                page = source["page_number"]
                place = f"{source['filename']}, p. {page}" if page else source["filename"]
                state = "cited" if source["cited"] else "retrieved, not cited"
                ui.quote(
                    source["text"],
                    f"{source['label']} · {place} · {state} · similarity {source['score']:.2f}",
                )


def _suggest(question: str) -> None:
    st.session_state[_PENDING_KEY] = question


def render(ctx: ui.AppContext) -> None:
    ui.page_header("Ask Documents", "Answers come only from your documents, with their sources.")
    documents = ctx.db.list_by_status(DocumentStatus.READY)
    if not documents:
        ui.no_documents()
        return
    ui.extractive_notice(ctx)

    names = {_ALL_DOCUMENTS: "All documents"}
    names |= {doc.document_id: doc.original_filename for doc in documents}
    ids = list(names)
    active = st.session_state.get(ui.ACTIVE_DOCUMENT_KEY)
    picker, new, clear = st.columns([4, 1.2, 1.2], vertical_alignment="bottom")
    selected = picker.selectbox(
        "Search in",
        ids,
        index=ids.index(active) if active in names else 0,
        format_func=names.get,
        key="chat_document",
    )
    document_id = selected or None

    conversation_id = ctx.db.latest_conversation(document_id)
    if new.button("New chat", icon=":material/add_comment:", width="stretch"):
        ctx.db.create_conversation(document_id)
        st.rerun()
    if clear.button(
        "Clear chat",
        icon=":material/delete_sweep:",
        width="stretch",
        disabled=conversation_id is None,
    ):
        ctx.db.delete_conversation(conversation_id)
        st.rerun()

    messages = ctx.db.list_messages(conversation_id) if conversation_id else []
    if not messages:
        ui.section_title("Try asking")
        for column, question in zip(st.columns(len(_SUGGESTIONS)), _SUGGESTIONS):
            column.button(
                question,
                width="stretch",
                key=f"suggest_{question}",
                on_click=_suggest,
                args=(question,),
            )
    for message in messages:
        with st.chat_message(message.role, avatar=ui.AVATARS[message.role]):
            if message.role == "user":
                st.markdown(ui.safe_markdown(message.content))
            else:
                _show_assistant(message.content, message.payload)

    typed = st.chat_input("Ask a question about your documents")
    question = typed or st.session_state.pop(_PENDING_KEY, None)
    if not question:
        return
    if conversation_id is None:
        conversation_id = ctx.db.create_conversation(document_id)
    with st.chat_message("user", avatar=ui.AVATARS["user"]):
        st.markdown(ui.safe_markdown(question))
    assistant = st.chat_message("assistant", avatar=ui.AVATARS["assistant"])
    with assistant, st.spinner("Reading the documents"):
        answer = ctx.services.answerer.ask(
            question,
            [document_id] if document_id else None,
            [m.content for m in messages if m.role == "user"][-_RECENT_QUESTIONS:],
            model=ctx.model,
            use_llm=ctx.model is not None,
        )
    ctx.db.add_message(conversation_id, "user", question)
    ctx.db.add_message(conversation_id, "assistant", answer.text, _payload(answer))
    where = names[selected]
    ctx.db.log_activity("question", f"Asked a question in {where}", document_id)
    st.rerun()
