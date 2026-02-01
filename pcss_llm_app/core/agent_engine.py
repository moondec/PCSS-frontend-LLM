from typing import List, Dict, Any
import re
import json
import datetime
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pcss_llm_app.core.tools import DocumentTools, OCRTools, PandocTools, VisionTools, WebSearchTools, ChartTools, FolderTools

class LangChainAgentEngine:
    def __init__(self, api_key: str, model_name: str, workspace_path: str, 
                 log_callback=None, custom_instructions: str = None):
        self.api_key = api_key
        self.model_name = model_name
        self.workspace_path = workspace_path
        self.log_callback = log_callback
        self.custom_instructions = custom_instructions or ""
        self.active_scratchpad = "" # Persistence layer for long tasks
        self._initialize_agent()

    def _log(self, message: str):
        if self.log_callback:
            self.log_callback(message)

    def _initialize_agent(self):
        # 1. Initialize LLM with performance optimizations
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url="https://llm.hpc.pcss.pl/v1",
            model=self.model_name,
            temperature=0,
            max_tokens=2048,  # Limit response length for faster generation
            request_timeout=120  # 2 minute timeout
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

        # Add Folder Tools
        folder_tools = FolderTools(root_dir=str(self.workspace_path))
        self.tools.extend(folder_tools.get_tools())

        # Add Pandoc Tools
        pandoc_tools = PandocTools(root_dir=str(self.workspace_path))
        self.tools.extend(pandoc_tools.get_tools())

        # Add Vision Tools (Hybrid Agent)
        vision_tools = VisionTools(root_dir=str(self.workspace_path), api_key=self.api_key, model_name=self.model_name)
        self.tools.extend(vision_tools.get_tools())

        # Add Web Search Tools
        web_search_tools = WebSearchTools(
            api_key=self.api_key, 
            model_name=self.model_name,
            base_url="https://llm.hpc.pcss.pl/v1"
        )
        self.tools.extend(web_search_tools.get_tools())

        # Add Chart Generation Tools
        chart_tools = ChartTools(root_dir=str(self.workspace_path))
        self.tools.extend(chart_tools.get_tools())



        # print("DEBUG: Building map", flush=True)
        self.tool_map = {t.name: t for t in self.tools}

    def run_step(self, prompt, stop=None):
         return self.llm.invoke(prompt, stop=stop).content

    # Intelligent Run method with loop detection and flexible limits
    # Intelligent Run method with loop detection and flexible limits
    def run(self, input_text: str, chat_history: List = None):
        if chat_history is None:
            chat_history = []

        tool_names = ", ".join(self.tool_map.keys())
        tool_descriptions = "\n".join([f"{t.name}: {t.description}" for t in self.tools])
        
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        system_template = f"""You are an AI assistant with tools. Date: {current_date}

Tools:
{tool_descriptions}

Format:
Question: [user's question]
Thought: [your reasoning]
Action: [one of: {tool_names}]
Action Input: [JSON for multi-arg tools, string for single-arg]
Observation: [result]
... (repeat as needed)
Thought: I have the answer
Final Answer: [your response]

Examples:
- convert_document: {{"source_path": "report.html", "output_format": "docx"}}
- save_document: {{"file_path": "doc.html", "content": "<h1>Title</h1><p>...</p>", "title": "Doc"}}
- write_file: {{"file_path": "notes.txt", "text": "Details..."}}
- list_directory: {{"dir_path": "."}}
- search_web: {{"query": "news Poland"}}

Rules:
- Use search_news for current events, search_web for general info
- Use visit_page to read full content from URLs (2-3 max)
- For documents: save_document with HTML content
- For simple text: write_file with 'file_path' and 'text'
- Be efficient - stop when you have enough information

{f"User Instructions: " + self.custom_instructions if self.custom_instructions else ""}
Begin!"""
        
        # Build conversation history
        history_text = ""
        for msg in chat_history:
            role = "Question" if isinstance(msg, HumanMessage) else "Final Answer" 
            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(msg, HumanMessage):
                history_text += f"Question: {content}\n"
            else:
                 history_text += f"Final Answer: {content}\n"

        # Stateful Continuation Logic
        is_continuation = any(word in input_text.lower() for word in ["kontynuuj", "continue", "zacznij od", "dalej"])
        
        if is_continuation and self.active_scratchpad:
            self._log("Resuming from previous scratchpad...")
            # Resume exactly where we left off
            prompt = f"{system_template}\n{history_text}\n(Resuming task...)\nThought: I should continue my work. Here is what I have done so far:\n{self.active_scratchpad}\nObservation: Please continue from the last point. Remember to use 'Action:' and 'Action Input:' for next steps.\nThought:"
        else:
            self.active_scratchpad = "" # Reset if it's a new task
            prompt = f"{system_template}\n{history_text}\nQuestion: {input_text}\nThought:"

        max_steps = 50
        action_history = []
        thought_history = []
        observation_history = []
        
        for i in range(max_steps):
            # Reset variables at start of each iteration to prevent stale values
            action = None
            action_input = None
            tool_args = None
            match = None
            
            self._log(f"--- Step {i+1} ---")
            
            # Invoke LLM with stop sequence
            self._log("Thinking...")
            response = self.llm.invoke(prompt, stop=["Observation:"])
            output = response.content
            print(f"--- Step {i} ---\nLLM Output:\n{output}\n----------------")
            self._log(f"Agent Thought:\n{output}")
            
            # Smart Loop Detection (Thoughts)
            # Normalize thought (remove whitespace/newlines for comparison)
            current_thought = output.replace("Thought:", "").strip()
            # If we have seen this exact thought recently (in last 3 steps), it's a loop
            if len(thought_history) > 0 and current_thought == thought_history[-1]:
                 self._log("⚠️ Thought Loop detected! Agent is repeating itself.")
                 return "Agent stopped: Repetitive thought process detected. The task seems completed or the agent is stuck."
            
            thought_history.append(current_thought)
            
            prompt += output
            self.active_scratchpad += output # Mirror to scratchpad
            
            if "Final Answer:" in output:
                self.active_scratchpad = "" # Task finished, clear scratchpad
                return output.split("Final Answer:")[-1].strip()
            
            # Parse Action
            pattern = r"Action:\s*(.+?)\nAction Input:\s*(.+)"
            match = re.search(pattern, output, re.DOTALL)
            
            # Fallback for "function call" style
            if not match:
                 json_pattern = r'function call\s*({.*?})'
                 json_match = re.search(json_pattern, output, re.DOTALL)
                 if json_match:
                     try:
                         func_data = json.loads(json_match.group(1))
                         if "name" in func_data:
                             action = func_data["name"]
                             args = func_data.get("arguments", {})
                             action_input = json.dumps(args) if isinstance(args, dict) else str(args)
                             match = True 
                     except Exception: pass

            if match:
                if hasattr(match, 'group'):
                    action = match.group(1).strip()
                    action_input = match.group(2).strip()
                
                # Safe Sanitization: stripping markdown wrappers only if they encompass the whole input
                action_input = action_input.strip()
                if "```" in action_input:
                    # Look for a block that starts at the beginning and ends at the end
                    block_match = re.search(r"^```(?:json)?\s*(.*?)\s*```$", action_input, re.DOTALL)
                    if block_match:
                        action_input = block_match.group(1).strip()
                    else:
                        # If not a whole-block wrap, only strip if it explicitly starts and ends with backticks
                        if action_input.startswith("```") and action_input.endswith("```"):
                             action_input = re.sub(r"^```(?:json)?", "", action_input)
                             action_input = re.sub(r"```$", "", action_input).strip()

                # Remove surrounding quotes only if they wrap the whole thing
                if (action_input.startswith('"') and action_input.endswith('"')) or \
                   (action_input.startswith("'") and action_input.endswith("'")):
                    action_input = action_input[1:-1].strip()
                
                # Try parsing as JSON
                try:
                    tool_args = json.loads(action_input)
                except json.JSONDecodeError:
                    # Robust extraction: finding the outer-most JSON object if parsing failed
                    # This handles cases like: {"key": "val"}?? or text ending with garbage
                    json_obj_match = re.search(r"(\{.*\})", action_input, re.DOTALL)
                    if json_obj_match:
                        try:
                            tool_args = json.loads(json_obj_match.group(1))
                        except:
                            tool_args = action_input
                    else:
                         tool_args = action_input

                # Argument Mapping Fallback
                if isinstance(tool_args, dict):
                    # Handle nested JSON string in a single field (Agent hallucination)
                    if len(tool_args) == 1:
                        key = list(tool_args.keys())[0]
                        val = tool_args[key]
                        if isinstance(val, str) and val.strip().startswith("{"):
                            try:
                                nested = json.loads(val)
                                if isinstance(nested, dict):
                                    tool_args = nested
                            except json.JSONDecodeError:
                                # Try fixing common JSON string errors (unescaped newlines)
                                try:
                                    fixed_val = val.replace('\n', '\\n').replace('\r', '')
                                    nested = json.loads(fixed_val)
                                    if isinstance(nested, dict):
                                        tool_args = nested
                                except: pass
                            except: pass

                    # Tool-specific Alias Mapping
                    if action == "list_directory":
                        if "path" in tool_args: tool_args["dir_path"] = tool_args.pop("path")
                        elif "file_path" in tool_args: tool_args["dir_path"] = tool_args.pop("file_path")
                    
                    elif action == "create_directory":
                        if "file_path" in tool_args: tool_args["directory_path"] = tool_args.pop("file_path")
                        elif "path" in tool_args: tool_args["directory_path"] = tool_args.pop("path")
                    
                    elif action in ["read_file", "write_file", "delete_file", "move_file", "copy_file", "read_docx", "read_pdf"]:
                        if "path" in tool_args and "file_path" not in tool_args:
                            tool_args["file_path"] = tool_args.pop("path")
                    
                    elif action == "convert_document":
                        if "path" in tool_args: tool_args["source_path"] = tool_args.pop("path")
                        elif "file_path" in tool_args: tool_args["source_path"] = tool_args.pop("file_path")

                    if action in ["write_file", "write_docx"] and "content" in tool_args and "text" not in tool_args:
                         tool_args["text"] = tool_args.pop("content")

                # Action Loop Detection
                current_action = (action, action_input)
                if action_history.count(current_action) >= 3: # Allow one retry
                    self._log("⚠️ Action Loop detected! Stopping agent.")
                    return "Agent stopped: Repetitive action loop. I have tried this too many times."
                
                action_history.append(current_action)

                # Execute Tool
                if action in self.tool_map:
                    self._log(f"Executing Tool: {action} (Step {i+1}/{max_steps})")
                    tool = self.tool_map[action]
                    try:
                        observation = tool.run(tool_args) if hasattr(tool, "run") else tool(tool_args)
                        
                        # Observation Loop / Stagnation check
                        if len(observation_history) > 0 and observation == observation_history[-1]:
                             self._log("⚠️ Stagnation detected (Repeated Observation).")
                             observation += " (Warning: You received this exact same result in the previous step. Stop if you are done.)"
                        
                        observation_history.append(observation)
                        self._log(f"Observation: {observation}")
                    except Exception as e:
                        observation = f"Error executing {action}: {e}"
                        self._log(f"Error: {observation}")
                else:
                    observation = f"Error: Tool '{action}' not found."
                
                obs_text = f"\nObservation: {observation}\nThought:"
                prompt += obs_text
                self.active_scratchpad += obs_text
            else:
                if "Action:" in output and "Action Input:" not in output:
                     prompt += "\nAction Input:"
                     self.active_scratchpad += "\nAction Input:"
                     continue
                if not output.strip(): return "Error: Agent produced empty response."
                fmt_error = "\nObservation: Invalid format. Please use 'Action:' and 'Action Input:'\nThought:"
                prompt += fmt_error
                self.active_scratchpad += fmt_error

        return f"Agent reached safety limit of {max_steps} steps without finishing. To prevent excessive API usage, I have stopped here. You can ask me to 'continue' if you believe more progress can be made."
