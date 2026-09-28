from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DocumentBlock:
    block_id: str
    kind: str
    location: str
    text: str
    section_path: tuple[str, ...] = ()


@dataclass
class DocumentArtifact:
    source_path: Path
    source_id: str
    blocks: list[DocumentBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if block.text)


class DocumentReader:
    """Read source documents into traceable blocks without semantic guessing."""

    def read(self, path: str | Path) -> DocumentArtifact:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Item Definition不存在: {source}")
        suffix = source.suffix.lower()
        if suffix == ".docx":
            blocks = self._read_docx(source)
        elif suffix == ".pdf":
            blocks = self._read_pdf(source)
        else:
            raise ValueError(f"不支持的Item Definition格式: {suffix}")
        if not blocks:
            raise ValueError(f"文档未提取到有效文本: {source}")
        return DocumentArtifact(source, source.name, blocks)

    @staticmethod
    def _read_docx(path: Path) -> list[DocumentBlock]:
        from docx import Document
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = Document(str(path))
        body = list(document.element.body.iterchildren())
        blocks: list[DocumentBlock] = []
        headings: dict[int, str] = {}
        paragraph_index = table_index = 0
        for body_index, element in enumerate(body):
            if element.tag == qn("w:p"):
                paragraph_index += 1
                paragraph = Paragraph(element, document._body)
                text = paragraph.text.strip()
                if not text:
                    continue
                level = DocumentReader._heading_level(paragraph, body, body_index)
                if level is not None:
                    headings = {
                        key: value for key, value in headings.items() if key < level
                    }
                    headings[level] = text
                blocks.append(DocumentBlock(
                    f"P-{paragraph_index:04d}", "paragraph",
                    f"paragraph[{paragraph_index}]", text,
                    tuple(headings[key] for key in sorted(headings)),
                ))
            elif element.tag == qn("w:tbl"):
                table_index += 1
                table = Table(element, document._body)
                section_path = tuple(headings[key] for key in sorted(headings))
                for row_index, row in enumerate(table.rows, start=1):
                    values = [cell.text.strip() for cell in row.cells]
                    text = " | ".join(value for value in values if value)
                    if text:
                        blocks.append(DocumentBlock(
                            f"T-{table_index:03d}-R-{row_index:04d}",
                            "table_row", f"table[{table_index}].row[{row_index}]",
                            text, section_path,
                        ))
        return blocks

    @staticmethod
    def _heading_level(paragraph, body: list, body_index: int) -> int | None:
        """Use Word headings, then short standalone headings lost to Normal style.

        Some supplied DOCX files style table section titles as Normal. A short
        bold paragraph is still a useful section boundary; a short unstyled
        umbrella title preceding one is treated as its parent. Neither a
        particular table number nor a domain-specific title is required.
        """

        text = paragraph.text.strip()
        style = str(paragraph.style.name or "")
        match = re.search(r"(?:heading|标题)\s*(\d+)", style, re.IGNORECASE)
        if match:
            return int(match.group(1))
        if len(text) > 60 or re.search(r"[。；;]$", text):
            return None
        is_bold = paragraph.style.font.bold is True or any(
            run.text.strip() and run.bold for run in paragraph.runs
        )
        if is_bold:
            return 3
        if len(text) > 30:
            return None
        from docx.oxml.ns import qn
        from docx.text.paragraph import Paragraph

        for next_element in body[body_index + 1:]:
            if next_element.tag == qn("w:tbl"):
                return 3
            if next_element.tag != qn("w:p"):
                continue
            following = Paragraph(next_element, paragraph._parent)
            if not following.text.strip():
                continue
            if any(run.text.strip() and run.bold for run in following.runs):
                return 2
            break
        return None

    @staticmethod
    def _read_pdf(path: Path) -> list[DocumentBlock]:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        blocks = []
        for index, page in enumerate(reader.pages, start=1):
            text = str(page.extract_text() or "").strip()
            if text:
                blocks.append(DocumentBlock(
                    f"PAGE-{index:04d}", "page", f"page[{index}]", text,
                ))
        return blocks
