try:
    from langgraph.prebuilt import create_react_agent
    print("create_react_agent found in langgraph.prebuilt")
except ImportError as e:
    print(f"Error: {e}")
