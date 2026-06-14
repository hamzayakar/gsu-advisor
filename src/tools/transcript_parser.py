import os
import json
import re
import logging
from pypdf import PdfReader
from openai import OpenAI
from src.config.settings import settings
from src.utils.observability import lf_manager

logger = logging.getLogger(__name__)

class TranscriptParser:
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.kloudeks_base_url,
            api_key=settings.kloudeks_api_key
        )
        
        parser_config = lf_manager.get_prompt_and_config("gsu_parser")
        self.model_name = parser_config["model_name"]
        self.parser_prompt = parser_config["system_prompt"]
        self.temperature = parser_config["temperature"]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
            return full_text
        except Exception as e:
            logger.error(f"Failed to read PDF: {e}")
            raise

    def parse_transcript(self, file_path: str) -> dict:
        file_ext = file_path.lower()
        
        # Desteklenen formatları kontrol et
        if not (file_ext.endswith('.pdf') or file_ext.endswith('.txt')):
            return {"error": "Unsupported format. Only PDF and TXT are supported."}

        try:
            if file_ext.endswith('.pdf'):
                raw_text = self.extract_text_from_pdf(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return {"error": f"File reading error: {str(e)}"}

        if not raw_text.strip():
            return {"error": "Empty text. Please upload a valid document containing readable text."}

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.parser_prompt},
                    {"role": "user", "content": f"TRANSCRIPT:\n{raw_text}"}
                ],
                temperature=self.temperature
            )
            
            raw_response = response.choices[0].message.content
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            
            if match:
                raw_json = json.loads(match.group())
                
                processed_courses = {}
                
                for course in raw_json.get("all_courses", []):
                    processed_courses[course["code"]] = course
                    
                failed_courses = []
                all_passed_courses = []
                
                for code, data in processed_courses.items():
                    if data["grade"] in ["F", "IA", "NP"]:
                        failed_courses.append({"code": code, "grade": data["grade"]})
                    else:
                        all_passed_courses.append({"code": code, "credits": data["credits"]})
                        
                final_json = {
                    "total_credits_completed": raw_json.get("total_credits_completed", 0.0),
                    "gpa": raw_json.get("gpa", 0.0),
                    "failed_courses": failed_courses,
                    "all_passed_courses": all_passed_courses
                }
                
                return final_json
                
            return {"error": "LLM did not return valid JSON."}
            
        except Exception as e:
            logger.error(f"Kloudeks API error: {e}")
            return {"error": str(e)}