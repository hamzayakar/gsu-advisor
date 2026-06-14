import gradio as gr
import hashlib
import logging
import os
from src.rag.agent import GSUAdvisorAgent
from src.tools.transcript_parser import TranscriptParser

# Configure logging to see what's happening in the terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize Core Components
# These will be loaded once when the app starts
agent = GSUAdvisorAgent()
parser = TranscriptParser()

_transcript_cache = {}
MAX_CACHE_SIZE = 50  # Limit the cache to 50 unique transcripts to prevent memory issues

def get_file_hash(filepath: str) -> str:
    """Generate a hash for the file content to identify unique uploads."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def chat_logic(user_message, history, file, request: gr.Request):
    """
    Main orchestration function for the Gradio interface.
    Integrates Transcript Parsing and Agentic RAG.
    """
    try:
        browser_session = request.session_hash if request else "local_dev"
        session_id = f"gsu_{hashlib.md5(browser_session.encode()).hexdigest()[:8]}"

        transcript_json = None
        if file is not None:
            # Generate a hash for the uploaded file to check for uniqueness
            file_hash = get_file_hash(file)
            
            if file_hash not in _transcript_cache:
                logger.info(f"New unique file upload detected. Parsing...")
                
                # RAM koruması: Cache çok şişerse sıfırla
                if len(_transcript_cache) >= MAX_CACHE_SIZE:
                    _transcript_cache.clear()
                    logger.info("Cache memory cleared to prevent leaks.")
                
                transcript_json = parser.parse_transcript(file)
                
                if "error" not in transcript_json:
                    _transcript_cache[file_hash] = transcript_json
                    logger.info("Transcript successfully structured and cached.")
                else:
                    logger.error(f"Parsing failed: {transcript_json['error']}")
            else:
                logger.info("File already parsed. Using cached transcript data.")
                transcript_json = _transcript_cache[file_hash]

        response = agent.ask(
            user_query=user_message,
            transcript_data=transcript_json,
            session_id=session_id,
            history=history
        )
        
        return response

    except Exception as e:
        logger.error(f"Critical UI Exception: {str(e)}")
        return f"Sistem hatası: {str(e)}. Lütfen yöneticinizle iletişime geçin."

# Define the Gradio Interface Layout
with gr.Blocks(title="GSÜ Sanal Akademik Danışman") as demo:
    gr.Markdown("# 🎓 GSÜ Sanal Akademik Danışman")
    gr.Markdown(
        "Galatasaray Üniversitesi Bilgisayar Mühendisliği mezuniyet koşulları ve akademik yönergeler hakkında "
        "bilgi alabilirsiniz. Daha isabetli bir analiz için transkriptinizi (PDF veya TXT) yüklemeyi unutmayın."
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Transkript Yükle (PDF veya TXT)",
                file_types=[".pdf", ".txt"],
                type="filepath"
            )
            gr.Info("Not: Transkript verileriniz sunucularımızda saklanmaz, sadece bu oturumda analiz edilir.")
            
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=chat_logic,
                additional_inputs=[file_input],
                examples=[
                    ["Mezuniyet için toplam kaç AKTS gerekiyor?"],
                    ["Transkriptimi incele, staj eksiğim var mı?"],
                    ["Bologna planına göre kaç teknik seçmeli ders almalıyım?"],
                    ["Merhaba, sistem nasıl çalışıyor?"]
                ],
                cache_examples=False
            )

    gr.Markdown("---")
    gr.Markdown("Developed for INF473 - Introduction to Generative AI | GSÜ BMB 2026")

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    from mcp_server import mcp

    app = FastAPI()
    app.mount("/mcp", mcp.sse_app())
    app = gr.mount_gradio_app(app, demo, path="/")

    uvicorn.run(app, host="0.0.0.0", port=7860)