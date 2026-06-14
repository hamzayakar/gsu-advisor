import logging
from mcp.server.fastmcp import FastMCP
from src.rag.retriever import GSURetriever

# Suppress excessive logs so they don't break MCP's JSON-RPC over stdio
logging.getLogger().setLevel(logging.WARNING)

# 1. Initialize the MCP Server
# This is the name that will be shown in the MCP network and can be used by other agents to discover and call tools.
mcp = FastMCP("GSU Academic Advisor MCP")

# 2. Initialize core tools
retriever = GSURetriever()

# 3. Expose tools to the MCP network using the @mcp.tool() decorator
@mcp.tool()
def search_graduation_rules(query: str) -> str:
    """
    Search the official Galatasaray University (GSU) Computer Engineering rulebase 
    for graduation requirements, internship rules, and ECTS limits.
    """
    results = retriever.retrieve(query=query, top_k=3)
    if not results:
        return "No relevant rules found in the database."
    
    context = ""
    for r in results:
        context += f"[Topic: {r['topic']}] {r['text']}\n"
    return context

if __name__ == "__main__":
    # Start the server (communicates via standard input/output)
    mcp.run()