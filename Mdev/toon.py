"""
toon.py
Token-Oriented Object Notation (TOON) Extractor Library
Wraps Docling and Pandas to produce structural JSON from PDFs.
"""

import json
import base64
import io
import pandas as pd
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import Table, TextItem, PictureItem, SectionHeaderItem

# --- Internal Helper Classes ---

class _ToonExtractor:
    def __init__(self):
        # Configure Docling pipeline
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.do_pictures = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def _img_to_b64(self, img_obj):
        if not img_obj: return ""
        try:
            buf = io.BytesIO()
            img_obj.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except: return ""

    def process(self, file_path):
        doc = self.converter.convert(file_path).document
        output = []
        
        # State Machine
        curr_page = -1
        curr_type = None
        buffer = []

        def flush():
            nonlocal curr_page, curr_type, buffer
            if not buffer: return
            
            # Formatting logic
            if curr_type == "text": content = "\n".join(buffer)
            elif curr_type == "table": content = "\n".join(buffer) # HTML tables
            elif curr_type == "image": content = buffer # List of b64
            else: content = ""

            output.append({
                "page_no": curr_page,
                "content_type": curr_type,
                "page_content": content
            })
            buffer = []

        for item, _ in doc.iterate_items():
            # 1. Get Page
            p = item.prov[0].page_no if (hasattr(item, "prov") and item.prov) else -1
            
            # 2. Get Type
            t = "unknown"
            if isinstance(item, Table): t = "table"
            elif isinstance(item, PictureItem): t = "image"
            elif isinstance(item, (TextItem, SectionHeaderItem)):
                if not item.text.strip(): continue
                t = "text"

            # 3. State Change Check
            if p != curr_page or t != curr_type:
                flush()
                curr_page = p
                curr_type = t

            # 4. Process Content
            if t == "table":
                df = item.export_to_dataframe()
                if df is not None: buffer.append(df.to_html(index=False, border=1))
            elif t == "image":
                img = item.get_image(doc)
                b64 = self._img_to_b64(img)
                if b64: buffer.append(f"data:image/png;base64,{b64}")
            elif t == "text":
                buffer.append(item.text)

        flush()
        return output

# --- Public API ---

def extract(file_path: str) -> list:
    """
    Parses a PDF and returns data in TOON format.
    Usage: data = toon.extract("file.pdf")
    """
    extractor = _ToonExtractor()
    return extractor.process(file_path)

def dump(data, file_path: str):
    """
    Saves TOON data to a JSON file.
    Usage: toon.dump(data, "output.json")
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def dumps(data) -> str:
    """
    Returns TOON data as a JSON string.
    Usage: json_str = toon.dumps(data)
    """
    return json.dumps(data, indent=4)