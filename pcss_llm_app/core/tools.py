import os
from typing import Optional
from langchain_core.tools import tool, StructuredTool
from docx import Document
from pypdf import PdfReader
from openai import OpenAI
import base64
import mimetypes

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
