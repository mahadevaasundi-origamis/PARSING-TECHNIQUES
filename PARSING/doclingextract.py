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

def extract_structured_json( file_path: str, MIN_WIDTH=200, MIN_HEIGHT=200, MIN_AREA=1000):
    
    # 1. Configuration
    try:
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
        
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        return []

    output_data = []
    
    # State variables for grouping
    current_page = -1
    current_type = None
    current_content_buffer = []

    # 2. Helper to flush the buffer
    def flush_buffer():
        if not current_content_buffer:
            return
        
        # Join content based on type
        if current_type == "text":
            final_content = "\n".join(current_content_buffer)
        elif current_type == "table":
            final_content = "\n".join(current_content_buffer)
        elif current_type == "image":
            final_content = current_content_buffer if len(current_content_buffer) > 1 else current_content_buffer[0]
        else:
            final_content = ""

        # Only add non-empty blocks
        if current_type == "text" and not final_content.strip():
            return
        if current_type == "image" and not current_content_buffer:
            return

        output_data.append({
            "page_no": current_page,
            "content_type": current_type,
            "page_content": final_content
        })

    # 4. Iterate through all items
    try:
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
                # Ignore empty whitespace items
                if not item.text.strip():
                    continue
                item_type = "text"
            else:
                continue  # Skip unknown types

            # --- Normal grouping logic ---
            # Trigger new block if page or type changes
            if item_page != current_page or item_type != current_type:
                flush_buffer()
                
                # Reset state
                current_page = item_page
                current_type = item_type
                current_content_buffer = []

            # --- Process Content by Type ---
            if item_type == "table":
                try:
                    df = item.export_to_dataframe()
                    if df is not None and not df.empty:
                        html_table = df.to_html(index=False, border=1)
                        current_content_buffer.append(html_table)
                    else:
                        print(f"⚠️ Empty table on page {item_page}, skipping")
                except Exception as e:
                    print(f"⚠️ Table processing error on page {item_page}: {str(e)}")
                    
            elif item_type == "image":
                try:
                    img_obj = item.get_image(doc)
                    if img_obj:
                        width, height = img_obj.size
                        
                        # Filter by size
                        if width >= MIN_WIDTH and height >= MIN_HEIGHT and (width * height) >= MIN_AREA:
                            
                            # Create base64 string
                            b64_str = image_to_base64(img_obj)
                            
                            img_data = {
                                "base64": f"data:image/png;base64,{b64_str}",
                                "width": width,
                                "height": height,
                                "page": item_page
                            }
                            current_content_buffer.append(img_data)
                            
                            print(f"✓ Extracted image: {width}x{height}px on page {item_page}")
                        else:
                            print(f"✗ Skipping small image: {width}x{height}px on page {item_page}")
                except Exception as e:
                    print(f"⚠️ Image processing error on page {item_page}: {str(e)}")
                    
            elif item_type == "text":
                current_content_buffer.append(item.text)

        # 5. Final flush
        flush_buffer()

        # 7. Sort by page number for consistent output
        output_data.sort(key=lambda x: x['page_no'])

    except Exception as e:
        print(f"⚠️ Error during document iteration: {str(e)}")
        # Return whatever was collected so far
        
    return output_data


# --- Execution ---
if __name__ == "__main__":
    pdf_path = r"/Users/santusahoo/Documents/DAGENT/4933ac75_KA_2425_3044751_20250209003420.pdf"

    # Save as JSON file
    output_file = "structured_response.json"
    
    # Configuration options
    result = extract_structured_json(pdf_path)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
        
    print(f"✅ Extraction Complete. Processed {len(result)} content blocks.")