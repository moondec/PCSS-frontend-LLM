import os
import sys

# Add current directory to path
sys.path.append(os.path.abspath("."))

from pcss_llm_app.config import ConfigManager
from pcss_llm_app.core.agent_engine import LangChainAgentEngine

def main():
    config = ConfigManager()
    api_key = config.get_api_key()
    workspace_path = config.get_workspace_path()
    model_name = "bielik_11b"
    
    print(f"Initializing Agent with model: {model_name}")
    
    def mock_logger(msg):
        print(f"[CONSOLE]: {msg}")

    try:
        agent = LangChainAgentEngine(api_key, model_name, workspace_path, log_callback=mock_logger)
        
        # Enable some internal printing by monkeypatching or assuming I will add it?
        # For now, just run and rely on the fact that I removed prints.
        # I actually need to see the failure. 
        # Since I cleaned up prints, this script won't show MUCH unless I modify engine again.
        # But wait, I can modify the engine to print to stdout for now as part of "implementing console" prep?
        
        # User query that caused failure
        query = "napisz zaproszenie na jutrzejszy bal wydziałowy o 20:00 w stołówce Uczelni. zapisz je do zaproszenie.docx (w formacie MS Word). Zaformatuj je ładnie. Urzyj ładnych ozdobnych czcionek"
        
        print(f"Running query: {query}")
        response = agent.run(query)
        print(f"Response: {response}")

    except Exception as e:
        print(f"Caught error: {e}")

if __name__ == "__main__":
    main()
