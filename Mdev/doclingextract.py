import base64
import io
import json
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import TableItem, TextItem, PictureItem, SectionHeaderItem

def image_to_base64(pil_image):
    """Helper to convert PIL Image to Base64 string for JSON output"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_structured_json(file_path: str):
    # 1. Configuration
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"Processing {file_path}...")
    doc = converter.convert(file_path).document

    output_data = []
    
    # State variables to track grouping
    current_page = -1
    current_type = None
    current_content_buffer = []

    # 2. Helper to flush the buffer into the output list
    def flush_buffer():
        if not current_content_buffer:
            return
        
        # Join content based on type
        if current_type == "text":
            final_content = "\n".join(current_content_buffer)
        elif current_type == "table":
            # Tables are already HTML strings in the buffer, just join them if multiple appear consecutively
            final_content = "\n".join(current_content_buffer)
        elif current_type == "image":
            # Images are stored as Base64 strings
            final_content = current_content_buffer # Keep as list of images or join if preferred
        else:
            final_content = ""

        output_data.append({
            "page_no": current_page,
            "content_type": current_type,
            "page_content": final_content
        })

    # 3. Iterate through all items (The Logic Block)
    for item, level in doc.iterate_items():
        
        # --- Determine Item Type ---
        item_type = "unknown"
        item_page = -1
        
        # Get Page Number safely
        if hasattr(item, "prov") and item.prov:
            item_page = item.prov[0].page_no
        
        if isinstance(item, TableItem):
            item_type = "table"
        elif isinstance(item, PictureItem):
            item_type = "image"
        elif isinstance(item, (TextItem, SectionHeaderItem)):
            # Ignore empty whitespace items to keep output clean
            if not item.text.strip():
                continue
            item_type = "text"

        # --- Condition: Check if we need to start a new block ---
        # Trigger new item if: Page changes OR Content Type changes
        if item_page != current_page or item_type != current_type:
            flush_buffer() # Save previous block
            
            # Reset state for the new block
            current_page = item_page
            current_type = item_type
            current_content_buffer = []

        # --- Process Content based on Type ---
        if item_type == "table":
            # Convert Table to HTML
            df = item.export_to_dataframe()
            if df is not None:
                html_table = df.to_html(index=False, border=1)
                current_content_buffer.append(html_table)
                
        elif item_type == "image":
            # Extract Image -> Convert to Base64
            # Docling allows getting the image object via get_image(doc)
            try:
                img_obj = item.get_image(doc)
                if img_obj:
                    b64_str = image_to_base64(img_obj)
                    current_content_buffer.append(f"data:image/png;base64,{b64_str}")
            except Exception as e:
                current_content_buffer.append(f"[Image Error: {str(e)}]")
                
        elif item_type == "text":
            # Append Markdown text
            current_content_buffer.append(item.text)

    # 4. Final flush for the last block of data
    flush_buffer()

    return output_data

# --- Execution ---
if __name__ == "__main__":
    pdf_path = r"C:\Users\MahadevaA\OneDrive - CXIO Technologies Pvt Ltd\Documents\GitHub\ReSearch\PARSING-TECHNIQUES\CRPL-1N60001074-CADPO110494.pdf" 
    
    result = extract_structured_json(pdf_path)
    
    # Save as JSON file
    with open("structured_response.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
        
    print(f"✅ Extraction Complete. Processed {len(result)} content blocks.")
    
    # Preview the first few blocks
    for block in result[:3]:
        print(f"\n--- Block (Page {block['page_no']} - {block['content_type']}) ---")
        print(block['page_content'][:200] + "...") # Print first 200 chars