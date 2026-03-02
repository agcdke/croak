"""Generate PDF summary reports using ReportLab."""
import io
from datetime import datetime
from typing import List, Dict, Any

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from loguru import logger


def generate_chat_summary_pdf(
    chat_history: List[Dict[str, str]],
    document_sources: List[Dict[str, Any]],
    title: str = "RAG Chat Summary Report",
) -> bytes:
    """
    Generate a PDF summary of a chat session with sources.

    Args:
        chat_history: List of {"question": str, "answer": str} dicts
        document_sources: List of source document metadata dicts
        title: Title of the report

    Returns:
        PDF bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=1 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#4a5568"),
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    section_header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#2d3748"),
        spaceBefore=16,
        spaceAfter=8,
        borderPad=4,
    )
    question_style = ParagraphStyle(
        "Question",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#2b6cb0"),
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=4,
        leftIndent=10,
    )
    answer_style = ParagraphStyle(
        "Answer",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#1a202c"),
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        leftIndent=10,
    )
    source_style = ParagraphStyle(
        "Source",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#718096"),
        leftIndent=10,
    )

    story = []

    # ── Header ──────────────────────────────────────────────
    story.append(Paragraph(title, title_style))
    story.append(
        Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3182ce")))
    story.append(Spacer(1, 0.2 * inch))

    # ── Summary Stats ────────────────────────────────────────
    story.append(Paragraph("Session Overview", section_header_style))
    stats_data = [
        ["Metric", "Value"],
        ["Total Questions", str(len(chat_history))],
        ["Documents Indexed", str(len(document_sources))],
        ["Report Date", datetime.now().strftime("%Y-%m-%d")],
    ]
    stats_table = Table(stats_data, colWidths=[2.5 * inch, 3 * inch])
    stats_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3182ce")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ebf8ff"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bee3f8")),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ])
    )
    story.append(stats_table)
    story.append(Spacer(1, 0.2 * inch))

    # ── Document Sources ─────────────────────────────────────
    if document_sources:
        story.append(Paragraph("Indexed Document Sources", section_header_style))
        source_data = [["File Name", "Type"]]
        for src in document_sources:
            source_data.append([
                src.get("file_name", "N/A"),
                src.get("source_type", "N/A").upper(),
            ])
        src_table = Table(source_data, colWidths=[3.5 * inch, 2 * inch])
        src_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7fafc"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(src_table)
        story.append(Spacer(1, 0.2 * inch))

    # ── Chat History ─────────────────────────────────────────
    if chat_history:
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
        story.append(Paragraph("Chat History", section_header_style))
        for i, entry in enumerate(chat_history, 1):
            story.append(
                Paragraph(f"Q{i}: {entry.get('question', '')}", question_style)
            )
            answer_text = entry.get("answer", "No answer recorded.")
            story.append(Paragraph(answer_text, answer_style))

            if entry.get("sources"):
                src_files = [
                    s.get("metadata", {}).get("file_name", "unknown")
                    for s in entry["sources"]
                ]
                story.append(
                    Paragraph(f"📄 Sources: {', '.join(set(src_files))}", source_style)
                )
            story.append(Spacer(1, 0.1 * inch))

    # ── Footer note ──────────────────────────────────────────
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(
        Paragraph(
            "Generated by PDF & Turtle Chatbot | Powered by LangChain + ChromaDB + Ollama",
            ParagraphStyle(
                "Footer",
                parent=styles["Normal"],
                fontSize=8,
                textColor=colors.HexColor("#a0aec0"),
                alignment=TA_CENTER,
                spaceBefore=8,
            ),
        )
    )

    doc.build(story)
    buffer.seek(0)
    pdf_bytes = buffer.getvalue()
    logger.info(f"Generated PDF report: {len(pdf_bytes)} bytes")
    return pdf_bytes


def generate_document_summary_pdf(
    file_name: str,
    source_type: str,
    num_chunks: int,
    sample_content: List[str],
) -> bytes:
    """Generate a single-document ingestion summary PDF."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Document Ingestion Report: {file_name}", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    info = [
        ["Property", "Value"],
        ["File Name", file_name],
        ["Document Type", source_type.upper()],
        ["Chunks Indexed", str(num_chunks)],
        ["Ingestion Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    t = Table(info, colWidths=[2 * inch, 4 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3182ce")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4f8"), colors.white]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2 * inch))

    if sample_content:
        story.append(Paragraph("Sample Content Preview", styles["Heading2"]))
        for i, sample in enumerate(sample_content[:3], 1):
            story.append(Paragraph(f"Chunk {i}:", styles["Heading3"]))
            story.append(Paragraph(sample[:500] + "...", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
