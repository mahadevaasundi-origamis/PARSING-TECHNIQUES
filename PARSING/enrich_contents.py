import json
import requests

# --- CONFIGURATION ---
INPUT_FILE = '/Users/santusahoo/Documents/DAGENT/text_and_image_structured.json'  # The output from the previous script
OUTPUT_FILE = 'enriched_rag_data.json'
OLLAMA_URL = "http://localhost:11434/api/generate"

# Models
MODEL_VISION = "llava:7b"
MODEL_TEXT = "qwen2.5:latest"

def call_ollama(model, prompt, image_base64=None):
    """
    Generic function to call Ollama API.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1, # Low temperature for factual extraction
            "num_ctx": 4096     # Ensure enough context for large tables/text
        }
    }

    if image_base64:
        # Ollama expects raw base64, need to strip headers if present
        # e.g., "data:image/png;base64,iVBOR..." -> "iVBOR..."
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        payload["images"] = [image_base64]

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get('response', '')
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama ({model}): {e}")
        return None

def process_text_content(text_content):
    """Generates structured text with headers using Qwen."""
    prompt = (
        "You are an expert technical writer. "
        "Reformat the following raw OCR text into a well-structured document. "
        "1. Fix line breaks and join broken sentences. "
        "2. Organize the content into logical sections. "
        "3. Use Markdown headers strictly in this format: '## HEADER 1 ##', '## HEADER 2 ##'. "
        "4. Do not summarize; keep all the information detailed. "
        "\n\nRAW TEXT:\n"
        f"{text_content}"
    )
    print("  -> Cleaning text with Qwen...")
    return call_ollama(MODEL_TEXT, prompt)

def process_image_content(base64_string):
    """Generates detailed image description using Llava."""
    prompt = (
        "Describe this image in extreme detail. "
        "If it is a machine, describe its parts, shape, and likely function. "
        "If it contains text, transcribe the key text visible. "
        "If it is a diagram, explain what it demonstrates."
    )
    print("  -> Analyzing image with Llava...")
    return call_ollama(MODEL_VISION, prompt, image_base64=base64_string)

def process_table_content(html_content):
    """Generates table summary using Qwen."""
    prompt = (
        "Analyze the following HTML table representing technical specifications. "
        "Provide a detailed textual summary of the data. "
        "Highlight key metrics, values, and features found in the table. "
        "\n\nHTML TABLE:\n"
        f"{html_content}"
    )
    print("  -> Summarizing table with Qwen...")
    return call_ollama(MODEL_TEXT, prompt)

def main():
    print(f"Reading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Input file not found. Please run the previous script first.")
        return

    total_items = len(data)
    print(f"Processing {total_items} items. This may take a while depending on your GPU...")

    for index, item in enumerate(data):
        page = item.get('page_no')
        c_type = item.get('content_type')
        content = item.get('page_content')

        print(f"\n[{index+1}/{total_items}] Processing Page {page} ({c_type})...")

        if c_type == 'text':
            cleaned = process_text_content(content)
            if cleaned:
                item['cleaned_text'] = cleaned

        elif c_type == 'image':
            description = process_image_content(content)
            if description:
                item['img_description'] = description

        elif c_type == 'table':
            summary = process_table_content(content)
            if summary:
                item['table_description'] = summary

    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print("Done.")

if __name__ == "__main__":
    main()