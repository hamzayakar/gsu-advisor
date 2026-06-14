import json
import os
from pathlib import Path
import yaml
from google import genai
from google.genai import types
from pydantic import BaseModel
from src.config.settings import settings

# Enforce strictly typed output schema using Pydantic
class RuleChunk(BaseModel):
    topic: str
    rule_text: str
    related_course_code: str
    rule_type: str

class GSURuleChunker:
    def __init__(self):
        # Initialize the modern Google GenAI Client
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = "gemini-2.5-pro"
        
        # Load externalized prompts
        prompts_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
        with open(prompts_path, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

    def chunk_document(self, file_path: str) -> list[dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        try:
            # Generate structured JSON using the new SDK syntax
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"{self.prompts['chunker_prompt']}\n\nTEXT:\n{raw_text}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[RuleChunk],
                    temperature=0.0,
                ),
            )
            
            raw_chunks = json.loads(response.text)
            formatted_chunks = []
            
            # Map the parsed JSON to our expected vector DB payload format
            for i, chunk in enumerate(raw_chunks):
                formatted_chunks.append({
                    "text": chunk.get("rule_text", ""),
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "chunk_index": i,
                        "topic": chunk.get("topic", ""),
                        "related_course_code": chunk.get("related_course_code", ""),
                        "rule_type": chunk.get("rule_type", "")
                    }
                })
            
            return formatted_chunks
        except Exception as e:
            raise RuntimeError(f"Gemini API Error during chunking: {e}")

if __name__ == "__main__":
    import json
    import os

    chunker = GSURuleChunker()
    input_dir = os.path.join(os.getcwd(), "data", "raw_docs")
    output_dir = os.path.join(os.getcwd(), "data", "processed_chunks")
    output_file = os.path.join(output_dir, "chunks.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_chunks = []
    
    print(f"[*] Starting chunking process for directory: {input_dir}")
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".html"):
            file_path = os.path.join(input_dir, filename)
            print(f"  -> Processing: {filename}...")
            
            try:
                chunks = chunker.chunk_document(file_path)
                all_chunks.extend(chunks)
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                    
                print(f"    [+] Saved {len(chunks)} new chunks. Total so far: {len(all_chunks)}")
                
            except Exception as e:
                print(f"  [-] Error processing {filename}: {e}")
                print("  [*] Continuing with the next file...")

    print(f"\n[+] Process finished. Final count: {len(all_chunks)} chunks saved to {output_file}")