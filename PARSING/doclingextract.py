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

def extract_structured_json(file_path: str, MIN_WIDTH=300, MIN_HEIGHT=300, MIN_AREA=50000):
    # 1. Configuration
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.generate_picture_images = True

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

    # Image filtering thresholds (adjust these as needed)
    MIN_WIDTH = MIN_WIDTH  # pixels
    MIN_HEIGHT = MIN_HEIGHT  # pixels
    MIN_AREA = MIN_AREA  # pixels^2

    # 2. Helper to flush the buffer into the output list
    def flush_buffer():
        if not current_content_buffer:
            return
        
        # Join content based on type
        if current_type == "text":
            final_content = "\n".join(current_content_buffer)
        elif current_type == "table":
            final_content = "\n".join(current_content_buffer)
        elif current_type == "image":
            final_content = current_content_buffer
        else:
            final_content = ""

        # Only add non-empty blocks
        if (current_type == "image" and not current_content_buffer):
            return

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
            # Extract Image -> Check Size -> Convert to Base64 if large enough
            try:
                img_obj = item.get_image(doc)
                if img_obj:
                    # Get image dimensions
                    width, height = img_obj.size
                    
                    # Filter: Only process images larger than threshold
                    if width >= MIN_WIDTH and height >= MIN_HEIGHT and (width * height) >= MIN_AREA:
                        import os
                        output_folder = "extracted_images"
                        os.makedirs(output_folder, exist_ok=True)

                        # Save image to disk for inspection
                        img_filename = f"page_{item_page}_image_{width}x{height}.png"
                        img_path = os.path.join(output_folder, img_filename)
                        img_obj.save(img_path)
                        print(f"✓ Saved image: {img_path}")

                        # Also add to JSON output
                        b64_str = image_to_base64(img_obj)
                        current_content_buffer.append(f"data:image/png;base64,{b64_str}")
                        print(f"✓ Including image: {width}x{height}px on page {item_page}")
                    else:
                        print(f"✗ Skipping small image: {width}x{height}px on page {item_page}")
            except Exception as e:
                print(f"Image processing error: {str(e)}")
                
        elif item_type == "text":
            # Append Markdown text
            current_content_buffer.append(item.text)

    # 4. Final flush for the last block of data
    flush_buffer()

    return output_data

# --- Execution ---
if __name__ == "__main__":
    pdf_path = r"/Users/santusahoo/Documents/DAGENT/IFB dishwasher.pdf" 
    
    result = extract_structured_json(pdf_path)
    
    # Save as JSON file
    with open("structured_response.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
        
    print(f"✅ Extraction Complete. Processed {len(result)} content blocks.")
    
    # # Preview the first few blocks
    # for block in result[:3]:
    #     print(f"\n--- Block (Page {block['page_no']} - {block['content_type']}) ---")
    #     print(block['page_content'][:200] + "...") # Print first 200 chars