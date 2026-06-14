import logging
from langfuse import Langfuse
from src.config.settings import settings

logger = logging.getLogger(__name__)

class LangfuseManager:
    def __init__(self):
        try:
            if not settings.langfuse_public_key or not settings.langfuse_secret_key:
                raise ValueError("Langfuse keys missing.")
                
            self.client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host
            )
            self.is_active = True
        except Exception as e:
            logger.warning(f"Langfuse offline/missing (Running in Fallback Mode): {e}")
            self.is_active = False

    def get_prompt_and_config(self, prompt_name: str) -> dict:
        if not self.is_active:
            return self._get_fallback(prompt_name)

        try:
            prompt_obj = self.client.get_prompt(prompt_name)
            config = prompt_obj.config

            if not config or "model" not in config or "temperature" not in config:
                return self._get_fallback(prompt_name)

            return {
                "system_prompt": prompt_obj.compile(),
                "model_name": config["model"],
                "temperature": config["temperature"]
            }
        except Exception as e:
            logger.error(f"Langfuse fetch failed for '{prompt_name}': {e}")
            return self._get_fallback(prompt_name)

    def _get_fallback(self, prompt_name: str) -> dict:
        """Local fallbacks from prompts.yaml if Langfuse is down."""
        from pathlib import Path
        import yaml
        
        prompts_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        
        if prompt_name == "gsu_router":
            return {
                "system_prompt": prompts["router_fallback"],
                "model_name": settings.router_model,
                "temperature": 0.0
            }
        elif prompt_name == "gsu_parser":
            return {
                "system_prompt": prompts["parser_fallback"],
                "model_name": settings.parser_model,
                "temperature": 0.0
            }
        return {
            "system_prompt": prompts["agent_fallback"],
            "model_name": settings.agent_model,
            "temperature": 0.3
        }

lf_manager = LangfuseManager()