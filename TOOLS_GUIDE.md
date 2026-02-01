# Agent Tools Guide

This document describes the tools available to the Autonomous Agent in the PCSS LLM Client. The agent automatically selects the best tool for the job.

## 📂 File Management
Basic file system operations within the workspace.
*   **list_directory**: Lists files and folders.
*   **read_file**: Reads proper contents of text files.
*   **write_file**: Creates or overwrites text files.
*   **copy_file / move_file / delete_file**: Standard file operations.

## 📄 Document Processing (Advanced)
Specialized tools for handling office documents.
*   **read_docx / read_pdf**: Extracts text from DOCX and PDF files.
*   **write_docx**: Creates simple Word documents (plain text only).
*   **save_document** ⭐ (Recommended):
    *   *Function:* Creates formatted PDF, DOCX, HTML, or TXT files from HTML-formatted content.
    *   *Usage:* Provide HTML content (h1, h2, p, ul, li, b, i tags) and the tool handles conversion.
    *   *Example:* `save_document({"file_path": "report.pdf", "content": "<h1>Title</h1><p>Content...</p>", "title": "My Report"})`
    *   *Requires:* Pandoc (external tool).
*   **convert_document**:
    *   *Function:* Converts files between formats (e.g., HTML -> DOCX, HTML -> PDF).
    *   *Usage Strategy:* To create a complex report, the Agent first writes an HTML file (with tables, headers, bold text) and then converts it to DOCX/PDF.

## 👁️ OCR & Scanning
Tools for extracting text from images.

*   **ocr_image** ⭐ (Recommended):
    *   *Model:* **Nanonets-OCR-s**.
    *   *Function:* Extracts text from scans, photos, and images. Essential for reading invoices, charts, or document photos.

*   **analyze_image** (Legacy/Disabled):
    *   *Note:* Currently **not available** on PCSS due to lack of multimodal models. Use `ocr_image` instead to extract text and then ask the agent to interpret it.

## 🤖 How to use?
Just ask the agent!
*   *"Read this invoice.png and tell me the total."* (Uses `ocr_image`)
*   *"Create a sales report PDF with a table."* (Uses `write_file` [html] -> `convert_document`)

## 🌐 Internet Access
Tools that connect the agent to the outside world.

*   **search_web**:
    *   *Function:* Searches DuckDuckGo for real-time information.
    *   *Privacy:* No user tracking, no API keys required.
    *   *Use Cases:* Checking weather, stock prices, latest news, or documentation for new libraries.
