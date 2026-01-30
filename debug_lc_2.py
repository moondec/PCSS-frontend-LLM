import langchain.agents
print(dir(langchain.agents))
try:
    from langchain.agents import AgentExecutor
    print("AgentExecutor found")
except ImportError:
    print("AgentExecutor NOT found")
