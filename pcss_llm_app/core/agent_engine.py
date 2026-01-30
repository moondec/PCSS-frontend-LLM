from typing import List, Dict, Any
import re
import json
import datetime
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pcss_llm_app.core.tools import DocumentTools, OCRTools, PandocTools, VisionTools, WebSearchTools

class LangChainAgentEngine:
    def __init__(self, api_key: str, model_name: str, workspace_path: str, log_callback=None):
        self.api_key = api_key
        self.model_name = model_name
        self.workspace_path = workspace_path
        self.log_callback = log_callback
        self._initialize_agent()

    def _log(self, message: str):
        if self.log_callback:
            self.log_callback(message)

    def _initialize_agent(self):
        # 1. Initialize LLM
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url="https://llm.hpc.pcss.pl/v1",
            model=self.model_name,
            temperature=0
        )

        # 2. Initialize Tools
        # print("DEBUG: Init FileToolkit", flush=True)
        toolkit = FileManagementToolkit(root_dir=str(self.workspace_path))

        self.tools = toolkit.get_tools()
        
        # Add Document Tools
        # print("DEBUG: Init DocumentTools", flush=True)
        doc_tools = DocumentTools(root_dir=str(self.workspace_path))
        # print("DEBUG: Getting DocumentTools", flush=True)
        new_tools = doc_tools.get_tools()
        # print(f"DEBUG: Got {len(new_tools)} doc tools", flush=True)
        self.tools.extend(new_tools)

        # Add OCR Tools
        ocr_tools = OCRTools(root_dir=str(self.workspace_path), api_key=self.api_key)
        self.tools.extend(ocr_tools.get_tools())

        # Add Pandoc Tools
        pandoc_tools = PandocTools(root_dir=str(self.workspace_path))
        self.tools.extend(pandoc_tools.get_tools())

        # Add Vision Tools (Hybrid Agent)
        vision_tools = VisionTools(root_dir=str(self.workspace_path), api_key=self.api_key)
        self.tools.extend(vision_tools.get_tools())

        # Add Web Search Tools
        web_search_tools = WebSearchTools()
        self.tools.extend(web_search_tools.get_tools())



        # print("DEBUG: Building map", flush=True)
        self.tool_map = {t.name: t for t in self.tools}

    def run(self, input_text: str, chat_history: List = None):
        if chat_history is None:
            chat_history = []
        
        # Construct ReAct Prompt
        tool_names = list(self.tool_map.keys())
        tool_descriptions = "\n".join([f"{t.name}: {t.description}" for t in self.tools])
        
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        system_prompt = f"""Answer the following questions as best you can. You have access to the following tools:

Current Date: {current_date}

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{', '.join(tool_names)}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""
        
        # Build conversation history
        history_text = ""
        for msg in chat_history:
            role = "User" if isinstance(msg, HumanMessage) else "AI"
            # Attempt to extract content properly
            content = msg.content if hasattr(msg, 'content') else str(msg)
            history_text += f"{role}: {content}\n"
            
        full_prompt = f"{system_prompt}\n{history_text}\nQuestion: {input_text}\nThought:"
        
        # Manual ReAct Loop
        max_steps = 10
        current_scratchpad = ""
        
        for _ in range(max_steps):
            # Prompt the LLM
            # We append the scratchpad (history of thoughts/actions in this turn)
            prompt_input = full_prompt + current_scratchpad
            
            response_msg = self.llm.invoke(prompt_input)
            response_text = response_msg.content
            
            current_scratchpad += response_text
            
            # Parse for Action
            # We look for "Action: <name>" and "Action Input: <input>"
            # We use Regex to find the LAST action in the chunk (though ideally model stops)
            
            # Simple parser: look for "Final Answer:"
            if "Final Answer:" in response_text:
                return response_text.split("Final Answer:")[-1].strip()
            
            # Look for Action
            action_match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*)", response_text, re.DOTALL)
            if action_match:
                action_name = action_match.group(1).strip()
                action_input = action_match.group(2).strip().split('\n')[0] # Take first line of input if multiline? Or relies on observation next?
                
                # Careful extracting input, sometimes it's multiline. 
                # Usually ReAct expects "Observation:" to interpret stop.
                # But here we got the text.
                
                # Let's refine extraction.
                # We need to find Action and Action Input that are NOT followed by Observation yet (since we are generating it)
                pass
            
            # Regex to find action and input
            # We look for the pattern, but we need to be careful about multiple actions (unlikely with stop triggers, but here we don't have stop triggers set on LLM maybe?)
            # ChatOpenAI usually doesn't stop unless we tell it. 
            # We should probably set stop words for LLM invoke ["Observation:"]
            
            # Let's re-invoke with stop words for safety in next step
            pass

        return "Agent stopped due to max steps."

    def run_step(self, prompt, stop=None):
         return self.llm.invoke(prompt, stop=stop).content

    # Revised Run method
    def run(self, input_text: str, chat_history: List = None):
        if chat_history is None:
            chat_history = []

        tool_names = ", ".join(self.tool_map.keys())
        tool_descriptions = "\n".join([f"{t.name}: {t.description}" for t in self.tools])
        
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        system_template = f"""Answer the following questions as best you can. You have access to the following tools:

Current Date: {current_date}

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action. If the action requires multiple arguments (like 'write_file', 'write_docx'), provide a JSON string.
Example for write_file: {{"file_path": "example.txt", "text": "Hello World"}}
Example for write_docx: {{"file_path": "report.docx", "text": "Title\\n\\nContent paragraph."}}
Example for ocr_image: {{"file_path": "scan.png"}}
Example for convert_document: {{"source_path": "report.html", "output_format": "docx"}}
Example for analyze_image: {{"file_path": "chart.png", "prompt": "What is the trend?"}}
Example for search_web: {{"query": "current weather in Poznan"}}
IMPORTANT: The user does NOT know what happened after your training data cutoff. You MUST use 'search_web' tools for any query about current events, news, or specific technical documentation.
IMPORTANT: For best results with complex documents (tables, headers), write the content to an HTML file first using write_file, then use convert_document to transform it to PDF or DOCX.
IMPORTANT: Use 'ocr_image' for simple text extraction. Use 'analyze_image' for understanding layouts, charts, or describing scenes.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""
        # History
        history_text = ""
        for msg in chat_history:
            role = "Question" if isinstance(msg, HumanMessage) else "Final Answer" 
            # Note: Mapping history to ReAct format is tricky. 
            # Better to just dump it as context or assume it was Q/A.
            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(msg, HumanMessage):
                history_text += f"Question: {content}\n"
            else:
                 history_text += f"Final Answer: {content}\n"

        prompt = f"{system_template}\n{history_text}\nQuestion: {input_text}\nThought:"
        
        max_steps = 15
        
        for i in range(max_steps):
            # print(f"DEBUG: Start Step {i}", flush=True)
            self._log(f"--- Step {i+1} ---")
            
            # Invoke LLM with stop sequence
            self._log("Thinking...")
            # print(f"--- Step {i} ---")
            # print(f"Prompt end:\n{prompt[-500:]}") 
            
            response = self.llm.invoke(prompt, stop=["Observation:"])
            output = response.content
            # print(f"LLM Output:\n{output}\n----------------")
            self._log(f"Agent Thought:\n{output}")
            
            prompt += output
            
            if "Final Answer:" in output:
                return output.split("Final Answer:")[-1].strip()
            
            # Parse Action
            # Regex for "Action: ... \nAction Input: ..."
            # We look for the last occurrence in the output (should be only one due to stop)
            pattern = r"Action:\s*(.+?)\nAction Input:\s*(.+)"
            match = re.search(pattern, output, re.DOTALL)
            
            # Fallback for "function call" style (common in some PCSS models)
            if not match:
                 # Look for: function call {"name": "toolsname", "arguments": {...}}
                 json_pattern = r'function call\s*({.*?})'
                 json_match = re.search(json_pattern, output, re.DOTALL)
                 if json_match:
                     try:
                         func_data = json.loads(json_match.group(1))
                         if "name" in func_data:
                             action = func_data["name"]
                             # Handle arguments: could be dict or string
                             args = func_data.get("arguments", {})
                             if isinstance(args, dict):
                                 action_input = json.dumps(args)
                             else:
                                 action_input = str(args)
                             match = True # Flag as found
                     except Exception:
                         pass

            if match:
                if 'action' not in locals(): # If standard ReAct match
                     action = match.group(1).strip()
                     action_input = match.group(2).strip()
                
                # Sanitize input: remove surrounding quotes if present
                
                # Sanitize input: remove surrounding quotes if present
                if (action_input.startswith('"') and action_input.endswith('"')) or \
                   (action_input.startswith("'") and action_input.endswith("'")):
                    action_input = action_input[1:-1]
                
                # Try parsing as JSON
                tool_args = action_input
                try:
                    tool_args = json.loads(action_input)
                except json.JSONDecodeError:
                    pass # treat as string

                # Argument Mapping Fallback
                if isinstance(tool_args, dict):
                    # Map 'path' -> 'file_path'
                    if "path" in tool_args and "file_path" not in tool_args:
                         tool_args["file_path"] = tool_args.pop("path")
                    # Map 'content' -> 'text'
                    if "content" in tool_args and "text" not in tool_args:
                         tool_args["text"] = tool_args.pop("content")

                # Execute Tool
                if action in self.tool_map:
                    # print(f"DEBUG: Found tool {action}", flush=True)
                    self._log(f"Executing Tool: {action}")
                    tool = self.tool_map[action]
                    try:
                        # print(f"DEBUG: Executing {action}. Tool Type: {type(tool)}", flush=True)
                        if hasattr(tool, "run"):
                            # print("DEBUG: Using .run()", flush=True)
                            observation = tool.run(tool_args)
                        else:
                            # print("DEBUG: Using __call__", flush=True)
                            observation = tool(tool_args)
                        self._log(f"Observation: {observation}")
                    except Exception as e:
                        # print(f"DEBUG: Exception in tool exec: {e}", flush=True)
                        observation = f"Error executing {action}: {e}"
                        self._log(f"Error: {observation}")
                else:
                    observation = f"Error: Tool '{action}' not found. Available tools: {tool_names}"
                
                # print(f"Observation: {observation}")
                # Append Observation
                prompt += f"\nObservation: {observation}\nThought:"
            else:
                # If no action found but no final answer, force it to think or just return
                if "Action:" in output and "Action Input:" not in output:
                     prompt += "\nAction Input:" # Prompt it to continue
                     continue
                
                if not output.strip():
                     # Empty response?
                     return "Error: Agent produced empty response."
                     
                # If it just rambles without Action, maybe it's done or confused.
                # Let's assume it failed to follow format.
                prompt += "\nObservation: Invalid format. Please use 'Action:' and 'Action Input:'\nThought:"

        return "Agent stopped: Max steps reached."
