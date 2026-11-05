"""Turn results into files for download. Nothing here leaves the computer."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import escape

import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core.document_comparator import NOT_CORRECTNESS_NOTE, Comparison, UniquePassage
from core.insights import DocumentInsights
from core.keyword_extractor import Keyword

# Spreadsheets run a cell that starts with one of these as a formula.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_UNSAFE_FILENAME = re.compile(r"[^\w.-]+")
_FONT = "Vera"
_FONT_BOLD = "VeraBd"
_MAX_PDF_TEXT = 700


def safe_filename(name: str) -> str:
    """Reduce a document name to characters that are safe in any download."""
    return _UNSAFE_FILENAME.sub("_", name).strip("._") or "document"


def _csv_cell(value: object) -> object:
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return f"'{value}"
    return value


def keywords_csv(keywords: Sequence[Keyword], keyphrases: Sequence[Keyword] = ()) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["term", "type", "tfidf_score", "count"])
    for kind, items in (("keyword", keywords), ("keyphrase", keyphrases)):
        for item in items:
            writer.writerow([_csv_cell(item.term), kind, f"{item.score:.4f}", item.count])
    return buffer.getvalue()


def insights_json(insights: DocumentInsights) -> str:
    return json.dumps(insights.to_dict(), ensure_ascii=False, indent=2)


def _register_fonts() -> None:
    """Use the Unicode fonts bundled with reportlab, because the built-in ones are Latin-1."""
    if _FONT in pdfmetrics.getRegisteredFontNames():
        return
    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont(_FONT, str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont(_FONT_BOLD, str(fonts / "VeraBd.ttf")))
    registerFontFamily(_FONT, normal=_FONT, bold=_FONT_BOLD, italic=_FONT, boldItalic=_FONT_BOLD)


def _clip(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) > _MAX_PDF_TEXT:
        flat = flat[:_MAX_PDF_TEXT].rsplit(" ", 1)[0] + "..."
    return escape(flat)


def _page(page: int | None) -> str:
    return f" (p. {page})" if page else ""


def _percent(value: float | None) -> str:
    return "Not available" if value is None else f"{value * 100:.0f}%"


def comparison_pdf(comparison: Comparison) -> bytes:
    """Build a PDF report. Document text is escaped, because Paragraph reads it as markup."""
    _register_fonts()
    base = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=base["BodyText"], fontName=_FONT, fontSize=9, leading=13)
    heading = ParagraphStyle(
        "Section", parent=base["Heading2"], fontName=_FONT_BOLD, fontSize=12, spaceBefore=10
    )
    title = ParagraphStyle("Title", parent=base["Title"], fontName=_FONT_BOLD, fontSize=16)
    name_a, name_b = escape(comparison.filename_a), escape(comparison.filename_b)
    scores = comparison.scores

    story: list = [
        Paragraph("Document comparison report", title),
        Paragraph(f"<b>Document A:</b> {name_a}<br/><b>Document B:</b> {name_b}", body),
        Spacer(1, 6),
        Paragraph(escape(NOT_CORRECTNESS_NOTE), body),
        Paragraph("Similarity scores", heading),
    ]
    table = Table(
        [
            ["Measure", "Score", "What it measures"],
            ["Overall", _percent(scores.overall), "Weighted semantic and lexical similarity"],
            ["Semantic", _percent(scores.semantic), "Cosine similarity of passage embeddings"],
            ["Lexical", _percent(scores.lexical), "TF-IDF cosine similarity of the texts"],
            ["Exact wording", _percent(scores.exact), "Overlap of five-word sequences"],
            ["A found in B", _percent(scores.coverage_a), "Share of A with related content in B"],
            ["B found in A", _percent(scores.coverage_b), "Share of B with related content in A"],
        ],
        colWidths=[35 * mm, 30 * mm, 100 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), _FONT, 9),
                ("FONT", (0, 0), (-1, 0), _FONT_BOLD, 9),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ]
        )
    )
    story.append(table)
    story += [Paragraph("Summary", heading), Paragraph(escape(comparison.summary), body)]
    if comparison.explanation:
        story += [
            Paragraph(f"Explanation by a local LLM ({escape(comparison.explanation_model)})", heading),
            Paragraph(_clip(comparison.explanation), body),
        ]

    def add_list(label: str, items: Sequence[str]) -> None:
        if items:
            story.append(Paragraph(label, heading))
            story.extend(Paragraph(f"- {item}", body) for item in items)

    add_list("Common topics", [escape(t.label) for t in comparison.common_topics])
    add_list(f"Topics mainly in {name_a}", [escape(t.label) for t in comparison.unique_topics_a])
    add_list(f"Topics mainly in {name_b}", [escape(t.label) for t in comparison.unique_topics_b])
    add_list(
        "Content differences",
        [
            f"<b>{escape(item.kind.value)}</b><br/>"
            f"A{_page(item.page_a)}: {_clip(item.text_a)}<br/>"
            f"B{_page(item.page_b)}: {_clip(item.text_b)}"
            for item in comparison.differences
        ],
    )
    add_list(
        "Similar passages",
        [
            f"<b>{escape(item.kind.value)}, similarity {item.score:.2f}</b><br/>"
            f"A{_page(item.page_a)}: {_clip(item.text_a)}<br/>"
            f"B{_page(item.page_b)}: {_clip(item.text_b)}"
            for item in comparison.similar_passages
        ],
    )

    def unique(items: Sequence[UniquePassage]) -> list[str]:
        return [f"{_clip(item.text)}{_page(item.page)}" for item in items]

    add_list(f"Only in {name_a}", unique(comparison.unique_passages_a))
    add_list(f"Only in {name_b}", unique(comparison.unique_passages_b))
    add_list("Notes", [escape(notice) for notice in comparison.notices])

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title="Document comparison report",
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    document.build(story)
    return buffer.getvalue()
