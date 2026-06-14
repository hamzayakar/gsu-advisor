import json
import logging
from openai import OpenAI
from langfuse import observe, propagate_attributes

from src.config.settings import settings
from src.rag.retriever import GSURetriever
from src.utils.observability import lf_manager

logger = logging.getLogger(__name__)

class GSUAdvisorAgent:
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.kloudeks_base_url,
            api_key=settings.kloudeks_api_key
        )
        self.retriever = GSURetriever()
        
        router_config = lf_manager.get_prompt_and_config("gsu_router")
        agent_config = lf_manager.get_prompt_and_config("gsu_agent")
        
        self.router_model = router_config["model_name"]
        self.router_prompt = router_config["system_prompt"]
        self.router_temp = router_config["temperature"]
        
        self.agent_model = agent_config["model_name"]
        self.agent_prompt = agent_config["system_prompt"]
        self.agent_temp = agent_config["temperature"]

        # --- THE TOOLBOX: Defining what the Agent can do ---
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_graduation_rules",
                    "description": "Search the official GSU rulebase (Qdrant Vector DB) for graduation requirements, internship rules, and ECTS limits.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string", 
                                "description": "The specific topic to search for, e.g., 'internship days', 'total ECTS', 'technical electives'"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_student_transcript",
                    "description": "Retrieve the parsed JSON data of the student's uploaded transcript to see their current progress.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]

    @observe(as_type="generation", name="router_decision")
    def route_query(self, user_query: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.router_model,
                messages=[
                    {"role": "system", "content": self.router_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=self.router_temp
            )
            raw_response = response.choices[0].message.content
            clean_json = raw_response.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_json)
        except Exception as e:
            logger.warning(f"Router failed, defaulting to RAG: {e}")
            return {"intent": "rag", "response": ""}

    @observe(as_type="span", name="tool_execution")
    def _execute_tool(self, tool_call, transcript_data) -> str:
        """Executes the mapped python function based on the LLM's tool call."""
        name = tool_call.function.name
        
        # 1. JSON Parse Error check
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            logger.error("LLM sent invalid JSON for tool arguments.")
            return "Error: Invalid JSON format. Please generate valid tool arguments."
            
        logger.info(f"[*] Agent executing tool: '{name}' with args: {args}")

        if name == "search_graduation_rules":
            # 2. Missing parameter check
            query = args.get("query")
            if not query:
                return "Error: Missing 'query' parameter. You must provide a search query."
                
            results = self.retriever.retrieve(query=query, top_k=10)
            if not results:
                return "No relevant rules found in the database."
            context = ""
            for r in results:
                context += f"[Topic: {r['topic']}] {r['text']}\n"
            return context
            
        elif name == "get_student_transcript":
            if not transcript_data:
                return '{"error": "Kullanıcı henüz transkript yüklemedi. Lütfen kullanıcıdan önce transkript dosyasını yüklemesini rica et."}'
            return json.dumps(transcript_data, ensure_ascii=False)
            
        return "Unknown tool."

    @observe(as_type="generation", name="agent_execution")
    def ask(self, user_query: str, transcript_data: dict = None, session_id: str = "local_test", history: list = None) -> str:
        if not user_query or not user_query.strip():
            return "Lütfen geçerli bir soru veya mesaj girin."

        if transcript_data is None:
            transcript_data = {}

        with propagate_attributes(session_id=session_id, tags=["gsu_advisor", "tool_calling"]):
            
            # 1. Routing Check
            route_decision = self.route_query(user_query)
            if route_decision.get("intent") == "chat":
                return route_decision.get("response", "Merhaba! Size nasıl yardımcı olabilirim?")

            # 2. Setup initial messages
            history_str = ""
            if history:
                # Limit history to last 10 messages to protect context window
                for msg in history[-10:]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    history_str += f"{role}: {content}\n"

            messages = [
                {"role": "system", "content": self.agent_prompt},
                {"role": "user", "content": f"CHAT HISTORY:\n{history_str}\n\nCURRENT QUERY: {user_query}"}
            ]

            # 3. The Agentic Loop (Max 4 iterations to prevent infinite loops)
            max_loops = 4
            for iteration in range(max_loops):
                try:
                    logger.info(f"Agent Loop Iteration: {iteration + 1}")
                    response = self.client.chat.completions.create(
                        model=self.agent_model,
                        messages=messages,
                        tools=self.tools,
                        temperature=self.agent_temp
                    )
                    
                    response_message = response.choices[0].message
                    
                    # If the model wants to call a tool
                    if response_message.tool_calls:
                        messages.append(response_message) # Add the assistant's tool call request
                        
                        # Execute all tools requested by the model
                        for tool_call in response_message.tool_calls:
                            tool_result = self._execute_tool(tool_call, transcript_data)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": str(tool_result)
                            })
                        # Loop continues to the next iteration to send tool results back to the LLM
                    else:
                        # Final answer generated without needing more tools
                        logger.info("Agent reached final answer.")
                        return response_message.content

                except Exception as e:
                    logger.error(f"Agent loop failed: {e}")
                    return "Sistemde geçici bir hata oluştu. Lütfen tekrar deneyin."
            
            return "Ajan çok fazla adım attığı için işlem güvenlik sebebiyle durduruldu."