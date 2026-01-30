import os
from typing import Optional
from langchain_core.tools import tool, StructuredTool
from docx import Document
from pypdf import PdfReader
from openai import OpenAI
import base64
import mimetypes
try:
    import pypandoc
except ImportError:
    pypandoc = None

from duckduckgo_search import DDGS

class DocumentTools:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def _get_full_path(self, file_path: str) -> str:
        """Resolve path relative to root_dir and ensure safety."""
        # Simple path join, in production needs traversal protection
        return os.path.join(self.root_dir, file_path)

    # @tool("write_docx")
    def write_docx(self, file_path: str, text: str):
        """
        Creates a new Word document (.docx) with the given text.
        Args:
            file_path: The name of the file to save (e.g., 'document.docx').
            text: The text content to write into the document.
        """
        try:
            full_path = self._get_full_path(file_path)
            doc = Document()
            for line in text.split('\n'):
                doc.add_paragraph(line)
            doc.save(full_path)
            return f"Successfully created DOCX file: {file_path}"
        except Exception as e:
            return f"Error writing DOCX file: {str(e)}"

    # @tool("read_docx")
    def read_docx(self, file_path: str) -> str:
        """
        Reads text content from a Word document (.docx).
        Args:
            file_path: The name of the file to read.
        """
        try:
            full_path = self._get_full_path(file_path)
            if not os.path.exists(full_path):
                return f"Error: File {file_path} not found."
            
            doc = Document(full_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
        except Exception as e:
            return f"Error reading DOCX file: {str(e)}"

    # @tool("read_pdf")
    def read_pdf(self, file_path: str) -> str:
        """
        Reads text content from a PDF file (.pdf).
        Args:
            file_path: The name of the file to read.
        """
        try:
            full_path = self._get_full_path(file_path)
            if not os.path.exists(full_path):
                return f"Error: File {file_path} not found."
            
            reader = PdfReader(full_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF file: {str(e)}"

    def get_tools(self):
        """Returns the list of bound tools."""
        # We need to bind self to the tool methods manually or re-create them?
        # The @tool decorator works on functions. If used on methods, self needs handling.
        # Better approach: Define these as unbound functions or use StructuredTool.from_function
        # wrapping the instance methods.
        
        # Let's use closure-based definition inside get_tools to capture self
        # Or better, just use the methods since I used @tool decorator?
        # Actually @tool on methods might be tricky with 'self' in signature for LangChain.
        
        # Simpler Pattern:
        from langchain_core.tools import StructuredTool

        return [
            StructuredTool.from_function(
                func=self.write_docx,
                name="write_docx",
                description="Creates a new Word document (.docx) with the given text."
            ),
            StructuredTool.from_function(
                func=self.read_docx,
                name="read_docx",
                description="Reads text content from a Word document (.docx)."
            ),
            StructuredTool.from_function(
                func=self.read_pdf,
                name="read_pdf",
                description="Reads text content from a PDF file (.pdf)."
            ),
        ]

class OCRTools:
    def __init__(self, root_dir: str, api_key: str):
        self.root_dir = root_dir
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://llm.hpc.pcss.pl/v1"
        )
        self.model = "Nanonets-OCR-s"

    def _get_full_path(self, file_path: str) -> str:
        return os.path.join(self.root_dir, file_path)

    def ocr_image(self, file_path: str) -> str:
        """
        Extracts text from an image file (PNG, JPG, JPEG) using OCR.
        Args:
            file_path: The path to the image file.
        """
        try:
            full_path = self._get_full_path(file_path)
            if not os.path.exists(full_path):
                return f"Error: File {file_path} not found."
            
            # Basic mime check
            mime_type, _ = mimetypes.guess_type(full_path)
            if not mime_type or not mime_type.startswith('image'):
                return f"Error: File {file_path} does not appear to be an image ({mime_type})."

            with open(full_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe the text in this image verbatim. Output ONLY the text."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error performing OCR: {str(e)}"

    def get_tools(self):
        return [
            StructuredTool.from_function(
                func=self.ocr_image,
                name="ocr_image",
                description="Extracts text from an image file (PNG, JPG) using OCR. Use this to read scanned documents or text in images."
            )
        ]

class PandocTools:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def _get_full_path(self, file_path: str) -> str:
         return os.path.join(self.root_dir, file_path)

    def convert_document(self, source_path: str, output_format: str) -> str:
        """
        Converts a document (e.g. HTML) to another format (e.g. DOCX, PDF) using Pandoc.
        Args:
            source_path: Path to the source file (e.g. 'report.html').
            output_format: Target format extension (e.g. 'docx', 'pdf').
        """
        if pypandoc is None:
            return "Error: pypandoc module is not installed."

        try:
            full_source = self._get_full_path(source_path)
            if not os.path.exists(full_source):
                 return f"Error: Source file {source_path} not found."
            
            # Construct output filename
            base_name = os.path.splitext(source_path)[0]
            target_filename = f"{base_name}.{output_format}"
            full_target = self._get_full_path(target_filename)
            
            output = pypandoc.convert_file(full_source, output_format, outputfile=full_target)
            return f"Successfully converted {source_path} to {target_filename}."
        except Exception as e:
             return f"Error converting document: {str(e)}"

    def get_tools(self):
        return [
            StructuredTool.from_function(
                func=self.convert_document,
                name="convert_document",
                description="Converts documents between formats (e.g. HTML to DOCX). Best used for creating formatted reports: 'Write content to .html then convert to .docx'."
            )
        ]


class VisionTools:
    def __init__(self, root_dir: str, api_key: str):
        self.root_dir = root_dir
        self.api_key = api_key
        # Always use GPT-4o for vision tasks, regardless of main agent model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://llm.hpc.pcss.pl/v1"
        )
        self.model = "gpt-4o"

    def _get_full_path(self, file_path: str) -> str:
        return os.path.join(self.root_dir, file_path)

    def analyze_image(self, file_path: str, prompt: str = "Describe this image in detail.") -> str:
        """
        Analyzes an image file using a Vision LLM (GPT-4o) to understand its content, layout, or extract data.
        Args:
            file_path: The name of the image file (e.g., 'chart.png').
            prompt: Question or instruction about the image (e.g., 'What is the trend in this chart?').
        """
        try:
            full_path = self._get_full_path(file_path)
            if not os.path.exists(full_path):
                return f"Error: File {file_path} not found."
            
            # Basic mime check
            mime_type, _ = mimetypes.guess_type(full_path)
            if not mime_type or not mime_type.startswith('image'):
                return f"Error: File {file_path} does not appear to be an image ({mime_type})."

            with open(full_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    def get_tools(self):
        return [
            StructuredTool.from_function(
                func=self.analyze_image,
                name="analyze_image",
                description="Analyzes an image using GPT-4o. Use this to descriptive scenes, understand charts, or analyze document layouts. Input: file_path and prompt."
            )
        ]


class WebSearchTools:
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o", base_url: str = "https://llm.hpc.pcss.pl/v1"):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.client = None
        if self.api_key:
             try:
                 self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
             except Exception:
                 pass

    def get_tools(self):
        return [
            StructuredTool.from_function(
                func=self.search_web,
                name="search_web",
                description="Performs a web search using DuckDuckGo. Use this to find current events, specific facts, or data not in your training set. Input: query string."
            )
        ]

    def _refine_query(self, query: str) -> str:
        """
        Uses LLM to refine the search query for better results.
        """
        if not self.client:
            return query
            
        try:
            prompt = f"""You are a Search Engine Expert. Transform the following user query into a highly effective DuckDuckGo search string.
User Query: "{query}"

Guidelines:
1. Extract core keywords.
2. If looking for a definition, DO NOT exclude dictionaries.
3. If looking for specific data (prices, weather), exclude generic sites like dictionaries using -site:operator.
4. Use standard search operators (site:, ", -) effectively.
5. Return ONLY the optimized query string, nothing else.

Optimized Query:"""
            
            response = self.client.chat.completions.create(
                model=self.model_name, # Use the configured model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.3
            )
            refined = response.choices[0].message.content.strip()
            # Remove quotes if present
            if (refined.startswith('"') and refined.endswith('"')):
                refined = refined[1:-1]
            return refined
        except Exception as e:
            print(f"Query refinement failed: {e}")
            return query

    def search_web(self, query: str) -> str:
        """
        Performs a web search using DuckDuckGo.
        Args:
            query: The search query (e.g., 'current weather in Poznan', 'latest AI news').
        """
        try:
            # Smart Refinement
            final_query = self._refine_query(query)
            print(f"DEBUG: Refined Query: '{query}' -> '{final_query}'")

            # Fallback Filter (if refinement failed or LLM didn't catch it)
            # Only apply if query contains common trigger words
            should_filter = any(w in query.lower() for w in ["aktualny", "znaczenie", "co to"])
            
            results = []
            with DDGS() as ddgs:
                # Get up to 10 results
                ddgs_gen = ddgs.text(final_query, region="pl-pl", max_results=10)
                if ddgs_gen:
                    for r in ddgs_gen:
                         link = r.get('href', '')
                         if should_filter:
                             if any(x in link for x in ['sjp.pwn.pl', 'synonim.net', 'wiktionary.org', 'diki.pl', 'dobryslownik.pl', 'synonimy.pl', 'sjp.pl']):
                                 continue
                         results.append(f"Title: {r.get('title')}\nLink: {link}\nSnippet: {r.get('body')}\n")
            
            if not results:
                return "No results found."
            
            return "\n---\n".join(results)
        except Exception as e:
            return f"Error searching web: {str(e)}"

    def get_tools(self):
        return [
            StructuredTool.from_function(
                func=self.search_web,
                name="search_web",
                description="Searches the Internet for current information. Use this to find facts, news, documentation, or events that happened after the model's training data cut-off."
            )
        ]
